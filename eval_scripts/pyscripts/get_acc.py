import argparse
import difflib
import re


def parse_wer_file(file_path: str) -> dict:

    results = {}
    current_uid = None
    current_lab = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith("UID:"):
                    current_uid = stripped_line.split(" ", 1)[1]

                elif stripped_line.startswith("lab:"):
                    current_lab = stripped_line.split(" ", 1)[1]

                elif stripped_line.startswith("rec:"):
                    parts = stripped_line.split(" ", 1)

                    if len(parts) > 1:
                        current_rec = parts[1]
                    else:
                        current_rec = ""

                    if current_uid and current_lab is not None:
                        results[current_uid] = (current_lab, current_rec)
                        current_uid = None
                        current_lab = None
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return {}
    except Exception as e:
        print(f"解析文件时发生未知错误: {e}")
        return {}

    return results


def validate_sub(text: str, a: str, b: str) -> tuple[int, str, str]:

    pattern = r"substitute '(.*?)' with '(.*?)'"
    match = re.search(pattern, text)
    if not match:
        return (0, a, b)
    original_word, replacement_word = match.groups()
    yy = replacement_word
    yy = " ".join(yy)

    # 在字符串 a 中搜索 yy 的下标区间
    try:
        x1 = a.find(yy)
        if x1 == -1:
            return (0, a, b)
        x2 = x1 + len(yy)

    except Exception as e:
        print(f"搜索字符串 a 时发生错误: {e}")
        return (0, a, b)

    # 判断字符串 b 中 x1-x2 区间的内容是否是 yy
    if len(b) < x2:
        return (0, a, b)

    substring_b = b[x1:x2]
    if substring_b != yy:
        return (0, a, b)
    a_processed = a[:x1] + a[x2:]
    b_processed = b[:x1] + b[x2:]

    return (1, a_processed, b_processed)


def validate_sub_index(text: str, a: str, b: str) -> tuple[int, str, str]:
    ref = "".join(a.split())
    hyp = "".join(b.split())
    pattern = r"from index\s+(\d+)\s+to index\s+(\d+)\s+with\s+'(.*?)'"
    match = re.search(pattern, text)
    if match:
        try:
            x_str, y_str, z_str = match.groups()
            x = int(x_str)
            y = int(y_str)
            z = z_str
            if hyp[x - 1 : y - 1] == z:
                return (1, ref[: x - 1] + ref[y:], hyp[: x - 1] + hyp[y:])
            else:
                return (0, ref[: x - 1] + ref[y:], hyp[: x - 1] + hyp[y:])
        except ValueError:
            print(f"警告: 匹配成功，但无法将 '{x_str}' 或 '{y_str}' 转换为整数。")
            return None
    else:
        return None


def validate_ins(instruction: str, a: str, b: str) -> tuple[int, str, str]:

    patterns = {
        "before_index": re.compile(r"insert\s+'(.*?)'\s+before .*? at index\s+(\d+)"),
        "after_index": re.compile(r"insert\s+'(.*?)'\s+after .*? at index\s+(\d+)"),
        "before_word": re.compile(r"insert\s+'(.*?)'\s+before .*? '(.+?)'"),
        "after_word": re.compile(r"insert\s+'(.*?)'\s+after .*? '(.+?)'"),
        "at_end": re.compile(r"insert\s+'(.*?)'\s+at the end"),
        "at_beginning": re.compile(r"insert\s+'(.*?)'\s+at the beginning"),
    }
    a = a.split(" ")
    a = "".join(a)
    b = b.split(" ")
    b = "".join(b)

    insertion_content = None
    start_index = -1
    end_index = -1

    match = patterns["before_index"].search(instruction)
    if match:
        content, index_str = match.groups()
        user_index = int(index_str)
        start_index = user_index - 1
        insertion_content = content
        end_index = start_index + len(content)
    elif match := patterns["after_index"].search(instruction):
        content, index_str = match.groups()
        user_index = int(index_str)
        start_index = user_index
        insertion_content = content
        end_index = start_index + len(content)
    elif match := patterns["before_word"].search(instruction):
        content, target_word = match.groups()
        insertion_content = content
        pos = a.find(target_word)
        if pos != -1:
            end_index = pos
            start_index = end_index - len(content)
        else:
            return (0, a, b)
    elif match := patterns["after_word"].search(instruction):
        content, target_word = match.groups()
        insertion_content = content
        pos = a.find(target_word)
        if pos != -1:
            start_index = pos + len(target_word)
            end_index = start_index + len(insertion_content)
        else:
            return (0, a, b)
    elif match := patterns["at_end"].search(instruction):
        content = match.group(1)
        insertion_content = content
        start_index = len(a) - len(insertion_content)
        end_index = len(a)
    elif match := patterns["at_beginning"].search(instruction):
        content = match.group(1)
        insertion_content = content
        start_index = 0
        end_index = len(insertion_content)

    if insertion_content is None:
        return (1, a, b)

    if (
        not (0 <= start_index < end_index <= len(a))
        or a[start_index:end_index] != insertion_content
    ):
        return (1, a, b)

    a_processed = a[:start_index] + a[end_index:]

    validation_success = True
    b_processed = b

    if not (0 <= start_index < end_index <= len(b)):
        validation_success = False
    elif b[start_index:end_index] != insertion_content:
        validation_success = False

    if validation_success:
        b_processed = b[:start_index] + b[end_index:]
    return (int(validation_success), a_processed, b_processed)


def validate_del(instruction: str, a: str, b: str) -> tuple[bool, str, str]:

    pattern_content = re.compile(r"delete\s+'(.*?)'")
    pattern_index = re.compile(r"delete.*?from index\s+(\d+)\s+to index\s+(\d+)")

    start_0based = -1
    end_0based = -1
    content_to_delete = ""

    match_content = pattern_content.search(instruction)
    if match_content:
        content_to_delete = match_content.group(1)
        start_0based = a.find(content_to_delete)
        if start_0based == -1:
            return (0, b)
        end_0based = start_0based + len(content_to_delete)

    elif match_index := pattern_index.search(instruction):
        start_1based, end_1based = map(int, match_index.groups())

        start_0based = start_1based - 1
        end_0based = end_1based

        if not (0 <= start_0based < end_0based <= len(a)):
            return (0, b)

        content_to_delete = a[start_0based:end_0based]
    else:
        return (0, b)
    validation_success: int
    b_processed: str

    if len(b) < end_0based:
        validation_success = 1
        b_processed = b
    elif b[start_0based:end_0based] == content_to_delete:
        validation_success = 0
        b_processed = b[:start_0based] + b[end_0based:]
    else:
        validation_success = 1
        b_processed = b

    return (validation_success, b_processed)


def process_del(input_path, ref_file, hyp_file, wav_asr_text):

    uid2asr = {}
    if wav_asr_text is not None:
        with open(wav_asr_text, "r", encoding="utf-8") as asr_file:
            for line in asr_file:
                row = line.strip().split(" ")
                uid2asr[row[0]] = row[1]

    acc = 0
    tot = 0
    with open(input_path) as fn, open(ref_file, "w", encoding="utf-8") as outfile1, open(
        hyp_file, "w", encoding="utf-8"
    ) as outfile2:
        for line in fn:
            data = {}
            d_uid, d_path, d_ins, d_asr_text, d_edited_text_label = line.strip().split("|")
            if d_uid == "file_name":
                continue
            else:
                data["uid"] = d_uid
                data["asr_text"] = d_asr_text
                data["asr_label"] = d_asr_text
                data["edited_text_label"] = d_edited_text_label
                data["instruction"] = d_ins
            uid = data["uid"]
            instruction = data["instruction"]
            # print(instruction, data['asr_label'], data['edited_text'])
            if wav_asr_text is not None and uid in uid2asr:
                res = validate_del(instruction, data["asr_label"], uid2asr[uid])
            else:
                # res = validate_del(instruction, data["asr_label"], data["edited_text"])
                print(f"{data} miss something")
                continue
            # print(res)
            if res[0] == 1:
                acc = acc + 1
            tot = tot + 1
            outfile1.write(f"{uid} {data['edited_text_label']}\n")
            outfile2.write(f"{uid} {res[1]}\n")
    print("acc = ", round(acc * 100.0 / tot, 2))


def process_log_file(input_path, wer_txt_file, ref_file, hyp_file, task):

    parsed_results = parse_wer_file(wer_txt_file)
    acc = 0
    tot = 0
    with open(input_path) as fn, open(ref_file, "w", encoding="utf-8") as outfile1, open(
        hyp_file, "w", encoding="utf-8"
    ) as outfile2:
        for line in fn:
            data = {}
            d_uid, d_path, d_ins, d_asr_text, d_edited_text_label = line.strip().split("|")
            if d_uid == "file_name":
                continue
            else:
                data["uid"] = d_uid
                data["asr_text"] = d_asr_text
                data["asr_label"] = d_asr_text
                data["edited_text_label"] = d_edited_text_label
                data["instruction"] = d_ins
            if data["uid"] in parsed_results:
                if task == "sub":
                    if "And substitute the characters or words" in data["instruction"]:
                        res = validate_sub_index(
                            data["instruction"],
                            parsed_results[data["uid"]][0],
                            parsed_results[data["uid"]][1],
                        )
                        # print("res = ", res)
                    elif "And substitute" in data["instruction"]:
                        res = validate_sub(
                            data["instruction"],
                            parsed_results[data["uid"]][0],
                            parsed_results[data["uid"]][1],
                        )
                elif task == "ins":
                    res = validate_ins(
                        data["instruction"],
                        parsed_results[data["uid"]][0],
                        parsed_results[data["uid"]][1],
                    )
                if res[0] == 1:
                    acc = acc + 1
                tot = tot + 1
                ref = "".join(res[1].split())
                hyp = "".join(res[2].split())
                outfile1.write(f"{data['uid']} {ref}\n")
                outfile2.write(f"{data['uid']} {hyp}\n")
    print("acc = ", round(acc * 100.0 / tot, 2))


def find_edited_region(before_text: str, after_text: str, lang: str) -> dict:
    if lang == "en":
        before_words = before_text.split()
        after_words = after_text.split()
    else:
        before_words = list(before_text)
        after_words = list(after_text)

    matcher = difflib.SequenceMatcher(None, before_words, after_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return {
                "type": tag,
                "before_indices": (i1, i2),
                "after_indices": (j1, j2),
                "deleted_words": before_words[i1:i2],
                "inserted_words": after_words[j1:j2],
            }

    return {"type": "equal", "before_indices": None, "after_indices": None}


def validate_sub_open(asr_text, edited_text_label, ref, hyp, lang):

    asr_text = asr_text.lower()
    edited_text_label = edited_text_label.lower()
    ref = ref.lower()
    hyp = hyp.lower()
    result = find_edited_region(asr_text, edited_text_label, lang)
    if result["type"] == "replace":
        before_start, before_end = result["before_indices"]
        deleted_str = " ".join(result["deleted_words"])
        inserted_str = " ".join(result["inserted_words"])
        if lang == "en":
            insert_word = len(inserted_str.split(" "))
        else:
            insert_word = len("".join(inserted_str.split(" ")))
        hyp_new = [word for word in hyp.split(" ") if word]
        if lang == "en":
            if edited_text_label.split(" ")[
                before_start : before_start + insert_word
            ] == inserted_str.split(" "):
                if hyp_new[before_start : before_start + insert_word] == inserted_str.split(" "):
                    ref_noedit = " ".join(
                        edited_text_label.split(" ")[:before_start]
                        + edited_text_label.split(" ")[before_start + insert_word :]
                    )
                    hyp_noedit = " ".join(
                        hyp_new[:before_start] + hyp_new[before_start + insert_word :]
                    )
                    return (1, ref_noedit, hyp_noedit)
        else:
            if edited_text_label[before_start : before_start + insert_word] == "".join(
                inserted_str.split(" ")
            ):
                if "".join(hyp_new[before_start : before_start + insert_word]) == "".join(
                    inserted_str.split(" ")
                ):
                    ref_noedit = (
                        edited_text_label[:before_start]
                        + edited_text_label[before_start + insert_word :]
                    )
                    hyp_noedit = "".join(
                        hyp_new[:before_start] + hyp_new[before_start + insert_word :]
                    )
                    return (1, ref_noedit, hyp_noedit)

    return (0, ref, hyp)


def validate_ins_open(asr_text, edited_text_label, ref, hyp, lang):

    asr_text = asr_text.lower()
    edited_text_label = edited_text_label.lower()
    ref = ref.lower()
    hyp = hyp.lower()
    result = find_edited_region(asr_text, edited_text_label, lang)
    if result["type"] == "insert":
        before_start, before_end = result["before_indices"]
        deleted_str = " ".join(result["deleted_words"])
        inserted_str = " ".join(result["inserted_words"])
        if lang == "en":
            insert_word = len(inserted_str.split(" "))
        else:
            insert_word = len("".join(inserted_str.split(" ")))
        hyp_new = [word for word in hyp.split(" ") if word]
        if lang == "en":
            if edited_text_label.split(" ")[
                before_start : before_start + insert_word
            ] == inserted_str.split(" "):
                if hyp_new[before_start : before_start + insert_word] == inserted_str.split(" "):
                    ref_noedit = asr_text.lower()
                    hyp_noedit = " ".join(
                        hyp_new[:before_start] + hyp_new[before_start + insert_word :]
                    )
                    return (1, ref_noedit, hyp_noedit)
        else:
            if edited_text_label[before_start : before_start + insert_word] == "".join(
                inserted_str.split(" ")
            ):
                if "".join(hyp_new[before_start : before_start + insert_word]) == "".join(
                    inserted_str.split(" ")
                ):
                    ref_noedit = asr_text
                    hyp_noedit = "".join(
                        hyp_new[:before_start] + hyp_new[before_start + insert_word :]
                    )
                    return (1, ref_noedit, hyp_noedit)
    return (0, ref, hyp)


def validate_del_open(asr_text, edited_text_label, ref, hyp, lang):

    asr_text = asr_text.lower()
    edited_text_label = edited_text_label.lower()
    ref = ref.lower()
    hyp = hyp.lower()
    result = find_edited_region(asr_text, edited_text_label, lang)
    if result["type"] == "delete":
        before_start, before_end = result["before_indices"]
        deleted_str = " ".join(result["deleted_words"])
        inserted_str = " ".join(result["inserted_words"])
        deleted_word = len(deleted_str.split(" "))
        if lang == "en":
            deleted_word = len(deleted_str.split(" "))
        else:
            deleted_word = len("".join(deleted_str.split(" ")))
        hyp_new = [word for word in hyp.split(" ") if word]
        if lang == "en":
            if asr_text.split(" ")[before_start : before_start + deleted_word] == deleted_str.split(
                " "
            ):
                if hyp_new[before_start : before_start + deleted_word] != deleted_str.split(" "):
                    ref_noedit = edited_text_label.lower()
                    hyp_noedit = hyp
                    return (1, ref_noedit, hyp_noedit)
                else:
                    ref_noedit = edited_text_label.lower()
                    hyp_noedit = " ".join(
                        hyp_new[:before_start] + hyp_new[before_start + deleted_word :]
                    )
                    return (0, ref_noedit, hyp_noedit)
        else:
            if asr_text[before_start : before_start + deleted_word] == "".join(
                deleted_str.split(" ")
            ):
                if "".join(hyp_new[before_start : before_start + deleted_word]) != "".join(
                    deleted_str.split(" ")
                ):
                    ref_noedit = edited_text_label
                    hyp_noedit = "".join(hyp.split(" "))
                    return (1, ref_noedit, hyp_noedit)
                else:
                    ref_noedit = edited_text_label
                    hyp_noedit = "".join(
                        hyp_new[:before_start] + hyp_new[before_start + deleted_word :]
                    )
                    return (0, ref_noedit, hyp_noedit)
    return (0, ref, hyp)


def process_en(input_path, wer_txt_file, ref_file, hyp_file, task, lang):

    parsed_results = parse_wer_file(wer_txt_file)
    acc = 0
    tot = 0
    with open(input_path) as fn, open(ref_file, "w", encoding="utf-8") as outfile1, open(
        hyp_file, "w", encoding="utf-8"
    ) as outfile2:
        for line in fn:
            data = {}
            d_uid, d_path, d_ins, d_asr_text, d_edited_text_label = line.strip().split("|")
            if d_uid == "file_name":
                continue
            else:
                data["uid"] = d_uid
                data["asr_text"] = d_asr_text
                data["asr_label"] = d_asr_text
                data["edited_text_label"] = d_edited_text_label
                data["instruction"] = d_ins
            if data["uid"] in parsed_results:
                if task == "sub":
                    res = validate_sub_open(
                        data["asr_text"],
                        data["edited_text_label"],
                        parsed_results[data["uid"]][0],
                        parsed_results[data["uid"]][1],
                        lang,
                    )
                elif task == "ins":
                    res = validate_ins_open(
                        data["asr_text"],
                        data["edited_text_label"],
                        parsed_results[data["uid"]][0],
                        parsed_results[data["uid"]][1],
                        lang,
                    )
                else:
                    res = validate_del_open(
                        data["asr_text"],
                        data["edited_text_label"],
                        parsed_results[data["uid"]][0],
                        parsed_results[data["uid"]][1],
                        lang,
                    )
                if res[0] == 1:
                    acc = acc + 1
                # else:
                #     print(data['asr_text'], data['edited_text_label'], parsed_results[data['uid']][0], parsed_results[data['uid']][1], lang)
                tot = tot + 1
                ref = " ".join(res[1].split())
                hyp = " ".join(res[2].split())
                outfile1.write(f"{data['uid']} {ref}\n")
                outfile2.write(f"{data['uid']} {hyp}\n")
    print("acc = ", round(acc * 100.0 / tot, 2))


def main():
    parser = argparse.ArgumentParser(description="解析日志并生成两个格式化的输出文件。")
    parser.add_argument("input_file", help="输入的推理结果文件。")
    parser.add_argument("task", help="任务类型")
    parser.add_argument("--wer_file", type=str, default=None)
    parser.add_argument("--ref_file", type=str, default=None)
    parser.add_argument("--hyp_file", type=str, default=None)
    parser.add_argument("--wav_asr_text", type=str, default=None)
    parser.add_argument("--eval_mode", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None)

    args = parser.parse_args()
    if args.eval_mode == "open":
        process_en(
            args.input_file, args.wer_file, args.ref_file, args.hyp_file, args.task, args.lang
        )
    else:
        if args.task == "del":
            process_del(args.input_file, args.ref_file, args.hyp_file, args.wav_asr_text)
        else:
            process_log_file(
                args.input_file, args.wer_file, args.ref_file, args.hyp_file, args.task
            )


if __name__ == "__main__":
    main()
