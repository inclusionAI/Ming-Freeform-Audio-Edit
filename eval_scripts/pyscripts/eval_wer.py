#!/usr/bin/python
# ****************************************************************#
# ScriptName: local/eval_wer.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2024-06-24 16:16
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-06-16 11:37
# Function:
# ***************************************************************#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import logging
import os
import pdb
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Union

import tqdm
from tqdm.contrib.concurrent import process_map

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%a %d %b %Y %H:%M:%S",
    force=True,
)

remove_tag = True
spacelist = [" ", "\t", "\r", "\n"]
# fmt: off
puncts = ["!", ",","?","、","。","！","，","；","？","：","「","」","︰","『","』","《","》", "|"]
# 语气词归并。在WER计算中每个list内的各词认为等价。
tone_words = [
    ["嗯", "啊", "哦", "噢", "喔"], # 表肯定时，音近“嗯”
    ["喂", "欸", "哎", "唉", "诶"], # 接起电话时，音近“喂”
    ["呃", "嗯", "啊"], # 思考时，音近“呃”
    ["了", "啦", "咧", "咯", "哩", "啰", "喽", "嘞"], # 句尾语气词，音近“了”
    ["啊", "呀", "哇", "吧", "嘛"], # 句尾语气词，音近“啊”
    ["呐", "呢", "啊", "哪"], # 句尾语气词，音近“呐”
    ["吧", "呗"], # 句尾语气词，音近“吧”
    ["呦", "哟"], # 句尾语气词，音近“哟”
]
# fmt: on


def characterize(string: str) -> List[str]:
    """文本处理wordize

    1. 删除无效符号，只保留中文、英文、数字
    2. wordize
        a. 中文按字符拆分
        b. 英文数字按空格或无效符号拆分
        c. 用`<>`框起来的标签作为整体保留
    """
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        # https://unicodebook.readthedocs.io/unicode.html#unicode-categories
        if cat1 == "Zs" or cat1 == "Cn" or char in spacelist:  # space or not assigned
            i += 1
            continue
        if cat1 == "Lo":  # letter-other
            res.append(char)
            i += 1
        else:
            # some input looks like: <unk><noise>, we want to separate it to two words.
            # 截取最小单元，最小单元从char开始，到sep结束
            sep = " "
            if char == "<":
                sep = ">"
            j = i + 1
            while j < len(string):
                c = string[j]
                if (
                    ord(c) >= 128 or (c in spacelist) or (c == sep)
                ):  # ord(c) >= 128表示c不是英文或数字或英文/数字标点符号
                    break
                j += 1
            if j < len(string) and string[j] == ">":
                j += 1
            res.append(string[i:j])
            i = j
    return res


def stripoff_tags(x: str) -> str:
    """删除用`<>`框起来的标签"""
    if not x:
        return ""
    chars = []
    i = 0
    T = len(x)
    while i < T:
        if x[i] == "<":
            while i < T and x[i] != ">":
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return "".join(chars)


def normalize(
    sentence: Union[List, str],
    ignore_words: set,
    case_sensitive: bool,
    split: Dict[str, List] = None,
):
    """对characterize得到的列表进行后处理

    sentence, ignore_words are both in unicode

    Args:
        sentence: 长字符串或wordize后得到的列表
        ignore_words: 删除字符
        case_sensitive: 是否区分大小写
        split: 指定分词，此处针对英文，e.g. {AlphaGo:["Alpha","Go"]}
    """
    new_sentence = []
    for token in sentence:
        x = token
        if not case_sensitive:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:
    def __init__(self, unify_tone_words=False):
        self.unify_tone_words = unify_tone_words
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost["cor"] = 0
        self.cost["sub"] = 1
        self.cost["del"] = 1
        self.cost["ins"] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, "")
        rec.insert(0, "")
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element["dist"] = 0
                element["error"] = "non"
            while len(row) < len(rec):
                row.append({"dist": 0, "error": "non"})
        for i in range(len(lab)):
            self.space[i][0]["dist"] = i
            self.space[i][0]["error"] = "del"
        for j in range(len(rec)):
            self.space[0][j]["dist"] = j
            self.space[0][j]["error"] = "ins"
        self.space[0][0]["error"] = "non"
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = "none"
                dist = self.space[i - 1][j]["dist"] + self.cost["del"]
                error = "del"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]["dist"] + self.cost["ins"]
                error = "ins"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token or (
                    self.unify_tone_words
                    and any(
                        [
                            lab_token in tone_cluster and rec_token in tone_cluster
                            for tone_cluster in tone_words
                        ]
                    )
                ):
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["cor"]
                    error = "cor"
                else:
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["sub"]
                    error = "sub"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]["dist"] = min_dist
                self.space[i][j]["error"] = min_error
        # Tracing back
        result = {
            "lab": [],
            "rec": [],
            "all": 0,
            "cor": 0,
            "sub": 0,
            "ins": 0,
            "del": 0,
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]["error"] == "cor":  # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["cor"] = self.data[lab[i]]["cor"] + 1
                    result["all"] = result["all"] + 1
                    result["cor"] = result["cor"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]["error"] == "sub":  # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["sub"] = self.data[lab[i]]["sub"] + 1
                    result["all"] = result["all"] + 1
                    result["sub"] = result["sub"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]["error"] == "del":  # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]["all"] = self.data[lab[i]]["all"] + 1
                    self.data[lab[i]]["del"] = self.data[lab[i]]["del"] + 1
                    result["all"] = result["all"] + 1
                    result["del"] = result["del"] + 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, "")
                i = i - 1
            elif self.space[i][j]["error"] == "ins":  # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]["ins"] = self.data[rec[j]]["ins"] + 1
                    result["ins"] = result["ins"] + 1
                result["lab"].insert(0, "")
                result["rec"].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]["error"] == "non":  # starting point
                break
            else:  # shouldn't reach here
                print(
                    "this should not happen , i = {i} , j = {j} , error = {error}".format(
                        i=i, j=j, error=self.space[i][j]["error"]
                    )
                )
        return result

    def overall(self):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in self.data:
            result["all"] = result["all"] + self.data[token]["all"]
            result["cor"] = result["cor"] + self.data[token]["cor"]
            result["sub"] = result["sub"] + self.data[token]["sub"]
            result["ins"] = result["ins"] + self.data[token]["ins"]
            result["del"] = result["del"] + self.data[token]["del"]
        return result

    def cluster(self, data):
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in data:
            if token in self.data:
                result["all"] = result["all"] + self.data[token]["all"]
                result["cor"] = result["cor"] + self.data[token]["cor"]
                result["sub"] = result["sub"] + self.data[token]["sub"]
                result["ins"] = result["ins"] + self.data[token]["ins"]
                result["del"] = result["del"] + self.data[token]["del"]
        return result

    def keys(self):
        return list(self.data.keys())


def width(string):
    """
    获取字符串的总宽度
    """
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def default_cluster(word):
    """返回`word`的语言类型

    包含不止一种类型，则返回Other
    """
    try:
        unicode_names = [unicodedata.name(char) for char in word]
    except:
        print(word)
        return "Other"
    for i in reversed(range(len(unicode_names))):
        if unicode_names[i].startswith("DIGIT"):  # 1
            unicode_names[i] = "Number"  # 'DIGIT'
        elif unicode_names[i].startswith("CJK UNIFIED IDEOGRAPH") or unicode_names[i].startswith(
            "CJK COMPATIBILITY IDEOGRAPH"
        ):
            # 明 / 郎
            unicode_names[i] = "Mandarin"  # 'CJK IDEOGRAPH'
        elif unicode_names[i].startswith("LATIN CAPITAL LETTER") or unicode_names[i].startswith(
            "LATIN SMALL LETTER"
        ):
            # A / a
            unicode_names[i] = "English"  # 'LATIN LETTER'
        elif unicode_names[i].startswith("HIRAGANA LETTER"):  # は こ め
            unicode_names[i] = "Japanese"  # 'GANA LETTER'
        elif (
            unicode_names[i].startswith("AMPERSAND")
            or unicode_names[i].startswith("APOSTROPHE")
            or unicode_names[i].startswith("COMMERCIAL AT")
            or unicode_names[i].startswith("DEGREE CELSIUS")
            or unicode_names[i].startswith("EQUALS SIGN")
            or unicode_names[i].startswith("FULL STOP")
            or unicode_names[i].startswith("HYPHEN-MINUS")
            or unicode_names[i].startswith("LOW LINE")
            or unicode_names[i].startswith("NUMBER SIGN")
            or unicode_names[i].startswith("PLUS SIGN")
            or unicode_names[i].startswith("SEMICOLON")
        ):
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            del unicode_names[i]
        else:
            return "Other"
    if len(unicode_names) == 0:
        return "Other"
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names) - 1):
        if unicode_names[i] != unicode_names[i + 1]:
            return "Other"
    return unicode_names[0]


def compute_recall(lab, rec, hotwords_pattern):
    recall = defaultdict(dict)
    n_all = Counter(hotwords_pattern.findall(lab))
    n_recall = Counter(hotwords_pattern.findall(rec))
    for w in n_all:
        recall[w]["n_all"] = n_all[w]
        recall[w]["n_recall"] = min(n_recall.get(w, 0), n_all[w])
    return recall


class Worker:
    def __init__(self, unify_tone_words, hotword_file):
        self.unify_tone_words = unify_tone_words
        self.hotwords_pattern = None
        if hotword_file is not None:
            if hotword_file is not None:
                hotword_list = []
                with open(hotword_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip("\n").split(" | ")
                        word = line_s[0].lower()
                        if len(line_s) < 2 or int(line_s[1]) > 0:
                            hotword_list.append(word)
                self.hotwords_pattern = re.compile(
                    "|".join([re.escape(word) for word in hotword_list])
                )

    def run(self, uid_ref_hyp):
        uid, lab, rec = uid_ref_hyp
        calculator = Calculator(self.unify_tone_words)
        result = calculator.calculate(lab, rec)
        if self.hotwords_pattern is not None:
            recall_stat = compute_recall(
                "".join(lab).lower(), "".join(rec).lower(), self.hotwords_pattern
            )
            result["hotword_stat"] = recall_stat
        return uid, result


def compute_wer(
    tokens_dict_ref, tokens_dict_pred, hotword_file=None, unify_tone_words=False, njobs=0
):
    uid_ref_hyps = [(uid, tokens_dict_ref[uid], tokens_dict_pred[uid]) for uid in tokens_dict_ref]

    worker = Worker(unify_tone_words, hotword_file)
    if njobs > 0:
        result_list = process_map(worker.run, uid_ref_hyps, chunksize=100, max_workers=njobs)
    else:
        result_list = [worker.run(uid_ref_hyp) for uid_ref_hyp in tqdm.tqdm(uid_ref_hyps)]

    num_sentences = 0
    num_sentence_errors = 0

    result_overall = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
    hotword_stat = {}
    for _, result in result_list:
        if len(result) == 0:
            continue
        result_overall["all"] += result["all"]
        result_overall["cor"] += result["cor"]
        result_overall["sub"] += result["sub"]
        result_overall["ins"] += result["ins"]
        result_overall["del"] += result["del"]
        if result["all"] != 0:
            num_sentences += 1
            if result["ins"] + result["sub"] + result["del"] > 0:
                num_sentence_errors += 1
        if "hotword_stat" in result:
            for word in result["hotword_stat"]:
                if word not in hotword_stat:
                    hotword_stat[word] = {"n_hotword": 0, "n_hotword_recall": 0}
                hotword_stat[word]["n_hotword"] += result["hotword_stat"][word]["n_all"]
                hotword_stat[word]["n_hotword_recall"] += result["hotword_stat"][word]["n_recall"]

    if result_overall["all"] != 0:
        wer = (
            float(result_overall["ins"] + result_overall["sub"] + result_overall["del"])
            * 100.0
            / result_overall["all"]
        )
    else:
        wer = 0.0

    if num_sentences != 0:
        ser = num_sentence_errors * 100 / num_sentences
    else:
        ser = 0.0

    return (
        wer,
        ser,
        result_overall["all"],
        result_overall["cor"],
        result_overall["sub"],
        result_overall["del"],
        result_overall["ins"],
        hotword_stat,
        result_list,
    )


def read_inputs(
    ref_file,
    hyp_file,
    tochar=True,
    use_whisper_normalizer=False,
    use_basic=False,
    allow_absent=False,
):
    if use_whisper_normalizer:
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

        if use_basic:
            normalizer = BasicTextNormalizer()
        else:
            normalizer = EnglishTextNormalizer()
    else:
        normalizer = None

    ignore_words = set()
    case_sensitive = False

    if not case_sensitive:
        ig = set([w.upper() for w in ignore_words])
        ignore_words = ig

    default_clusters = {}
    default_words = {}

    text_dict_pred = {}
    text_dict_pred_norm = {}
    tokens_dict_pred = {}
    with codecs.open(hyp_file, "r", "utf-8") as fh:
        for line in fh:
            lsp = line.strip().split(maxsplit=1)
            if len(lsp) == 0:
                continue
            elif len(lsp) == 1:
                uid = lsp[0]
                line = ""
            else:
                uid, line = lsp
            if tochar:
                array = characterize(line)
            else:
                array = line.strip().split()
            if len(array) == 0:
                array = []
            # uid = array[0]
            text_dict_pred[uid] = "".join(array)
            if use_whisper_normalizer:
                assert normalizer is not None
                tokens_dict_pred[uid] = normalizer(" ".join(array)).strip().split()
            else:
                tokens_dict_pred[uid] = normalize(array, ignore_words, case_sensitive)
            text_dict_pred_norm[uid] = "".join(tokens_dict_pred[uid])

    text_dict_ref = {}
    text_dict_ref_norm = {}
    tokens_dict_ref = {}
    for line in open(ref_file, "r", encoding="utf-8"):
        lsp = line.strip().split(maxsplit=1)
        if len(lsp) == 0:
            continue
        elif len(lsp) == 1:
            uid = lsp[0]
            line = ""
        else:
            uid, line = lsp
        if tochar:
            array = characterize(line)
        else:
            array = line.rstrip("\n").split()
        if len(array) == 0:
            array = []
        # uid = array[0]
        if allow_absent and uid not in tokens_dict_pred:
            continue
        text_dict_ref[uid] = "".join(array)
        if use_whisper_normalizer:
            assert normalizer is not None
            lab = normalizer(" ".join(array)).strip().split()
        else:
            lab = normalize(array, ignore_words, case_sensitive)
        tokens_dict_ref[uid] = lab
        text_dict_ref_norm[uid] = "".join(lab)

        if uid not in tokens_dict_pred and args.ignore_unfound_sample:
            tokens_dict_pred[uid] = [""]
        # logging.info(f"{uid}, {line}")
        try:
            for word in tokens_dict_pred[uid] + lab:
                if word == "\x14\x17" or word == "":
                    continue
                if word not in default_words:
                    default_cluster_name = default_cluster(word)
                    if default_cluster_name not in default_clusters:
                        default_clusters[default_cluster_name] = {}
                    if word not in default_clusters[default_cluster_name]:
                        default_clusters[default_cluster_name][word] = 1
                    default_words[word] = default_cluster_name
        except:
            print(uid)

    return (
        text_dict_pred,
        text_dict_pred_norm,
        tokens_dict_pred,
        text_dict_ref,
        text_dict_ref_norm,
        tokens_dict_ref,
    )


def main(args):
    if args.infer_file is not None:
        ref_file = "x1"
        hyp_file = "x2"
        with open(args.infer_file) as fn, open(ref_file, "w", encoding="utf-8") as outfile1, open(
            hyp_file, "w", encoding="utf-8"
        ) as outfile2:
            for line in fn:
                data = json.loads(line)
                outfile1.write(f"{data['uid']} {data['edited_text_label']}\n")
                outfile2.write(f"{data['uid']} {data['edited_text']}\n")
    else:
        ref_file = args.ref_file
        hyp_file = args.hyp_file
    if not os.path.exists(hyp_file):
        return
    (
        text_dict_pred,
        text_dict_pred_norm,
        tokens_dict_pred,
        text_dict_ref,
        text_dict_ref_norm,
        tokens_dict_ref,
    ) = read_inputs(
        ref_file,
        hyp_file,
        tochar=args.tochar,
        use_whisper_normalizer=args.use_whisper_normalizer,
        use_basic=args.use_basic,
        allow_absent=args.allow_absent,
    )

    (
        wer,
        ser,
        num_words,
        correct,
        substraction,
        deletion,
        insertion,
        hotword_stat,
        result_list,
    ) = compute_wer(
        tokens_dict_ref=tokens_dict_ref,
        tokens_dict_pred=tokens_dict_pred,
        hotword_file=args.hotword_file,
        unify_tone_words=args.unify_tone_words,
        njobs=args.njobs,
    )

    if args.verbose or args.verbose_excel:
        padding_symbol = " "
        max_words_per_line = sys.maxsize
        records = []

        text_extra = {}
        if args.excel_extra_text is not None:
            assert len(args.excel_extra_text) % 2 == 0
            for i in range(0, len(args.excel_extra_text), 2):
                key, filename = args.excel_extra_text[i : i + 2]
                text_extra[key] = {}
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip().split()
                        uid = line_s[0]
                        text = " ".join(line_s[1:])
                        text_extra[key][uid] = text

        for uid, result in result_list:
            record = {}
            if len(result) == 0:
                continue

            if args.verbose:
                print(f"UID: {uid}")
            if args.verbose_excel:
                record["uid"] = uid

            if result["all"] != 0:
                wer_utt = (
                    float(result["ins"] + result["sub"] + result["del"]) * 100.0 / result["all"]
                )
            else:
                wer_utt = 0.0

            if args.verbose:
                print(
                    f'WER: {wer_utt:4.2f}% N={result["all"]} C={result["cor"]} S={result["sub"]} D={result["del"]} I={result["ins"]}',
                    end="",
                )
            if args.verbose_excel:
                record["wer"] = wer_utt
                record["all"] = result["all"]
                record["cor"] = result["cor"]
                record["sub"] = result["sub"]
                record["del"] = result["del"]
                record["ins"] = result["ins"]
                record["ref"] = text_dict_ref[uid]
                record["ref_norm"] = text_dict_ref_norm[uid]
                record["pred"] = text_dict_pred[uid]
                record["pred_norm"] = text_dict_pred_norm[uid]
                for key in text_extra:
                    if uid in text_extra[key]:
                        record[key] = text_extra[key][uid]
            if "hotword_stat" in result:
                hotword_info_list = []
                for w in result["hotword_stat"]:
                    hotword_info_list.append(
                        f'{w}: {result["hotword_stat"][w]["n_recall"]}/{result["hotword_stat"][w]["n_all"]}'
                    )
                if args.verbose:
                    print(" HOTWORDS_RECALL: " + ", ".join(hotword_info_list), end="")
                if args.verbose_excel:
                    record["hotword"] = ", ".join(hotword_info_list)

            if args.verbose:
                print("")
                space = {}
                space["lab"] = []
                space["rec"] = []
                for idx in range(len(result["lab"])):
                    len_lab = width(result["lab"][idx])
                    len_rec = width(result["rec"][idx])
                    length = max(len_lab, len_rec)
                    space["lab"].append(length - len_lab)
                    space["rec"].append(length - len_rec)
                upper_lab = len(result["lab"])
                upper_rec = len(result["rec"])
                lab1, rec1 = 0, 0
                while lab1 < upper_lab or rec1 < upper_rec:
                    line_lab = ["lab:"]
                    line_rec = ["rec:"]
                    lab2 = min(upper_lab, lab1 + max_words_per_line)
                    for idx in range(lab1, lab2):
                        token = result["lab"][idx]
                        line_lab.append(token + (padding_symbol * space["lab"][idx]))
                    print(" ".join(line_lab))
                    rec2 = min(upper_rec, rec1 + max_words_per_line)
                    for idx in range(rec1, rec2):
                        token = result["rec"][idx]
                        line_rec.append(token + (padding_symbol * space["rec"][idx]))
                    print(" ".join(line_rec))
                    lab1 = lab2
                    rec1 = rec2
                print("")
            if args.verbose_excel:
                records.append(record)

        if args.verbose:
            print("\n\n")
        if args.verbose_excel is not None:
            import pandas as pd
            from pandas.io.formats import excel

            excel.ExcelFormatter.header_style = None
            df = pd.DataFrame(records)
            num_rows = len(df)
            df.at[num_rows, "wer"] = wer
            df.at[num_rows, "all"] = num_words
            df.at[num_rows, "cor"] = correct
            df.at[num_rows, "sub"] = substraction
            df.at[num_rows, "del"] = deletion
            df.at[num_rows, "ins"] = insertion
            df.to_excel(args.verbose_excel, index=None)

    result_str_list = [
        f"WER = {wer:0.2f}",
        f"SER = {ser:0.2f}",
        f"All = {num_words}",
        f"C = {correct}({correct / num_words * 100:0.2f}%)",
        f"S = {substraction}({substraction / num_words * 100:0.2f}%)",
        f"D = {deletion}({deletion / num_words * 100:0.2f}%)",
        f"I = {insertion}({insertion / num_words * 100:0.2f}%)",
    ]

    if len(hotword_stat) > 0:
        n_hotword_recall = sum([hotword_stat[w]["n_hotword_recall"] for w in hotword_stat])
        n_hotword = max(1, sum([hotword_stat[w]["n_hotword"] for w in hotword_stat]))
        result_str_list.append(
            f"Hotword recall = {n_hotword_recall / n_hotword * 100:0.2f}% = {n_hotword_recall}/{n_hotword}"
        )

    print(", ".join(result_str_list))

    if args.verbose_hotwords:
        print("\n\nHotword details:")
        for w in hotword_stat:
            if hotword_stat[w]["n_hotword"] > 0:
                print(
                    f'{w}: {hotword_stat[w]["n_hotword_recall"] / hotword_stat[w]["n_hotword"] * 100:0.2f}% = {hotword_stat[w]["n_hotword_recall"]}/{hotword_stat[w]["n_hotword"]}'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_file")
    parser.add_argument("hyp_file")
    parser.add_argument("--infer_file", type=str, default=None)
    parser.add_argument("--tochar", action="store_true")
    parser.add_argument("--unify_tone_words", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose_excel", type=str, default=None)
    parser.add_argument("--excel_extra_text", type=str, nargs="+", default=None)
    parser.add_argument("--hotword_file", type=str, default=None)
    parser.add_argument("--verbose_hotwords", action="store_true")
    parser.add_argument("--allow_absent", action="store_true")
    parser.add_argument("--use_whisper_normalizer", action="store_true")
    parser.add_argument("--use_basic", action="store_true")
    parser.add_argument("--njobs", type=int, default=0)
    parser.add_argument(
        "--ignore_unfound_sample",
        action="store_true",
        help="If true, set sample text to [] which unfound in hyp",
    )

    args = parser.parse_args()

    main(args)
