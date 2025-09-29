# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/9/11
import argparse
import os

import soundfile
import tqdm


def read_one_meta_file(ts_dir, ts_wav_dir, lang):
    if os.path.exists(f"{ts_dir}/{lang}/tag.time_stretch.dur"):
        with open(f"{ts_dir}/{lang}/tag.time_stretch.dur", "r") as fr:
            utt_dur_tag = [ii.split() for ii in fr.read().splitlines()]
        utt_dur_tag = [[ii[0], float(ii[1]), float(ii[2])] for ii in utt_dur_tag]
    else:
        with open(f"{ts_dir}/{lang}/meta.lst", "r") as fr:
            meta_lines = fr.read().splitlines()
        sav_lines, utt_dur_tag = [], []
        for meta_line in meta_lines:
            meta_line = meta_line.split("|")
            assert len(meta_line) == 5
            if not (
                meta_line[4].startswith("time_stretch(")
                or meta_line[2].startswith("adjusts the speed to")
            ):
                continue
            utt = meta_line[0]
            if meta_line[4].startswith("time_stretch("):
                prompt_args = float(meta_line[4].split("(")[1])
            else:
                prompt_args = float(meta_line[2].rsplit(maxsplit=1)[1])
            pro_utt = utt[len("time_stretch_") :] if utt.startswith("time_stretch_") else utt
            wav_data, sr = soundfile.read(f"../../wavs/{pro_utt}.wav")
            dur = wav_data.shape[0] / sr
            sav_lines.append(f"{utt} {dur} {dur/prompt_args}\n")
            utt_dur_tag.append([utt, dur, dur / prompt_args])
        with open(f"{ts_dir}/{lang}/tag.time_stretch.dur", "w") as fw:
            fw.write("".join(sav_lines))
    return utt_dur_tag


def ana_time_stretch(task_file, verbose_dir, ts_dir, ts_wav_dir, rank=0, num_proc=1):
    with open(task_file, "r") as fr:
        task_lines = fr.read().splitlines()
    os.makedirs(verbose_dir, exist_ok=True)
    utt_dur_tag_zh = read_one_meta_file(ts_dir, ts_wav_dir, "zh")
    utt_dur_tag_en = read_one_meta_file(ts_dir, ts_wav_dir, "en")
    for task_line in tqdm.tqdm(task_lines[rank::num_proc], disable=rank != 0):
        mdl_n, _, wav_dir = task_line.split()
        for lang, utt_dur_tag in [["zh", utt_dur_tag_zh], ["en", utt_dur_tag_en]]:
            res_s = []
            for utt, dur, tag in utt_dur_tag:
                wav_data, sr = soundfile.read(f"{wav_dir}/{utt}.wav")
                gen_dur = wav_data.shape[0] / sr
                res_s.append([utt, abs(gen_dur - tag), abs(gen_dur - tag) / dur])
            res_m1 = sum(ii[1] for ii in res_s) / len(res_s)
            res_m2 = sum(ii[2] for ii in res_s) / len(res_s)
            sav_lines = [f"{ii[0]} {ii[1]} {ii[2]}\n" for ii in res_s]
            sav_lines.append(f"delta {res_m1} {res_m2}\n")
            with open(f"{verbose_dir}/{mdl_n}.{lang}.dur.time_stretch", "w") as fw:
                fw.write("".join(sav_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, default="config/task.lst")
    parser.add_argument("--verbose_dir", type=str, default="logs_time_stretch_dur")
    parser.add_argument("--ts_dir", type=str, default="time_stretch")
    parser.add_argument(
        "--ts_wav_dir", type=str, default="/nativemm/share/cpfs/wanren.pj/eval/seedtts_testset"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    ana_time_stretch(
        args.task_file, args.verbose_dir, args.ts_dir, args.ts_wav_dir, args.rank, args.num_proc
    )


if __name__ == "__main__":
    main()
