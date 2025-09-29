# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/7/17
import argparse
import os
import string

import librosa
from zhon.hanzi import punctuation

punctuation_all = punctuation + string.punctuation
import jiwer
import soundfile
import tqdm
import zhconv
from funasr import AutoModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def process_one(hypo, truth, lang="zh"):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = jiwer.process_words(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures.wer
    subs = measures.substitutions / len(ref_list)
    dele = measures.deletions / len(ref_list)
    inse = measures.insertions / len(ref_list)
    return raw_truth, raw_hypo, wer, subs, dele, inse


def read_one_meta_file(ts_dir, lan):
    with open(f"{ts_dir}/{lan}/meta.lst", "r") as fr:
        meta_lines = fr.read().splitlines()
    utt2tag_txt = {}
    for meta_line in meta_lines:
        meta_line = meta_line.split("|")
        if meta_line[0] == "file_name":
            continue
        if len(meta_line) >= 4:
            utt, tag_txt = meta_line[0], meta_line[3]
        else:
            raise NotImplementedError
        assert utt not in utt2tag_txt
        utt2tag_txt[utt] = tag_txt
    return utt2tag_txt


class AsrZh:
    def __init__(self, mdl_p, ts_dir, bat_size, device):
        self.model = AutoModel(model=mdl_p, disable_update=True, device=device)  # device
        self.utt2tag_txt, self.bat_size = read_one_meta_file(ts_dir, "zh"), bat_size

    def __call__(self, wav_dir):
        asr_res_s = self.model.generate(
            input=[f"{wav_dir}/{utt}.wav" for utt in self.utt2tag_txt.keys()],
            batch_size=self.bat_size,
            disable_pbar=True,
        )
        return [[asr_res["key"], zhconv.convert(asr_res["text"], "zh-cn")] for asr_res in asr_res_s]


class AsrEn:
    def __init__(self, mdl_p, ts_dir, bat_size, device):
        self.processor = WhisperProcessor.from_pretrained(mdl_p)
        self.model = WhisperForConditionalGeneration.from_pretrained(mdl_p).to(device)
        self.utt2tag_txt, self.bat_size = read_one_meta_file(ts_dir, "en"), bat_size
        self.device = device

    def __call__(self, wav_dir):
        utt_wav = []
        for utt in self.utt2tag_txt.keys():
            wav, sr = soundfile.read(f"{wav_dir}/{utt}.wav")
            if sr != 16000:
                wav_to_resample = wav.T if wav.ndim > 1 else wav
                resampled_wav = librosa.resample(wav_to_resample, orig_sr=sr, target_sr=16000)
                wav = resampled_wav.T if resampled_wav.ndim > 1 else resampled_wav
                sr = 16000
            assert sr == 16000, f"sr = {sr}"
            utt_wav.append([utt, wav])
        utt_wav.sort(key=lambda lam_tmp: lam_tmp[1].shape[0])
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        asr_res_s = []
        for ii in range(0, len(utt_wav), self.bat_size):
            cur_bat = utt_wav[ii : ii + self.bat_size]
            input_features = self.processor(
                [wav[1] for wav in cur_bat], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)
            predicted_ids = self.model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            asr_res_s.extend([[cur_bat[bid][0], transcription[bid]] for bid in range(len(cur_bat))])
        return asr_res_s


def asr_wer(
    task_file,
    ts_dir,
    zh_mdl_p,
    en_mdl_p,
    bat_size=512,
    verbose_dir="",
    rank=0,
    num_proc=1,
    dialect=False,
):
    agent_zh = AsrZh(zh_mdl_p, ts_dir, bat_size, f"cuda:{rank}")
    if not dialect:
        agent_en = AsrEn(en_mdl_p, ts_dir, bat_size, f"cuda:{rank}")

    with open(task_file, "r") as fr:
        task_lines = fr.read().splitlines()

    if verbose_dir:
        os.makedirs(verbose_dir, exist_ok=True)

    if rank == 0:
        task_iter = tqdm.tqdm(task_lines[rank::num_proc])
    else:
        task_iter = task_lines[rank::num_proc]
    for task_line in task_iter:
        mdl_n, _, wav_dir = task_line.split()
        if not dialect:
            agent_list = [("zh", agent_zh), ("en", agent_en)]
        else:
            agent_list = [("zh", agent_zh)]
        for lan, agent in agent_list:
            verbose_lines, m_wer, m_ins, m_del, m_sub = [], [], [], [], []
            for utt, trans in agent(wav_dir):
                tag_txt = agent.utt2tag_txt[utt]
                raw_truth, raw_hypo, wer, subs, dele, inse = process_one(trans, tag_txt, lang=lan)
                verbose_lines.append(
                    f"{utt}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n"
                )
                m_wer.append(wer)
                m_ins.append(inse)
                m_del.append(dele)
                m_sub.append(subs)
            m_wer = f"{sum(m_wer) / len(m_wer):.4%}"
            m_sub = f"{sum(m_sub) / len(m_sub):.4%}"
            m_del = f"{sum(m_del) / len(m_del):.4%}"
            m_ins = f"{sum(m_ins) / len(m_ins):.4%}"
            verbose_lines.append("mdl,WER,SUB,DEL,INS\n")
            verbose_lines.append(f"{mdl_n},{m_wer},{m_sub},{m_del},{m_ins}\n")
            if verbose_dir:
                with open(f"{verbose_dir}/{mdl_n}_{lan}.log", "w") as fw:
                    fw.write("".join(verbose_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, default="config/task.lst")
    parser.add_argument("--verbose_dir", type=str, default="logs_wer")
    parser.add_argument(
        "--ts_dir",
        type=str,
        default="resource/seedtts_testset",
    )
    parser.add_argument(
        "--zh_mdl_p",
        type=str,
        default="Models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument(
        "--en_mdl_p",
        type=str,
        default="Models/openai-whisper-large-v3",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--dialect", type=bool, default=False)
    args = parser.parse_args()
    asr_wer(
        args.task_file,
        args.ts_dir,
        args.zh_mdl_p,
        args.en_mdl_p,
        args.batch_size,
        args.verbose_dir,
        args.rank,
        args.num_proc,
        args.dialect,
    )


if __name__ == "__main__":
    main()
