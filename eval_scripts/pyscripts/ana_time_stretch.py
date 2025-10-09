import argparse
import os

import soundfile
import tqdm


def read_one_meta_file(res_dir, lang):
    if os.path.exists(f"{res_dir}/tag.time_stretch.dur"):
        with open(f"{res_dir}/tag.time_stretch.dur", "r") as fr:
            utt_dur_tag = [ii.split() for ii in fr.read().splitlines()]
        utt_dur_tag = [[ii[0], float(ii[1]), float(ii[2])] for ii in utt_dur_tag]
    else:
        with open(f"../meta/time_stretch/meta_{lang}_time_stretch.csv", "r") as fr:
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
            wav_data, sr = soundfile.read(f"../wavs/{pro_utt}.wav")
            dur = wav_data.shape[0] / sr
            sav_lines.append(f"{utt} {dur} {dur/prompt_args}\n")
            utt_dur_tag.append([utt, dur, dur / prompt_args])
        with open(f"{res_dir}/tag.time_stretch.dur", "w") as fw:
            fw.write("".join(sav_lines))
    return utt_dur_tag


def ana_time_stretch(wav_dir, res_dir, lang):
    os.makedirs(res_dir, exist_ok=True)
    utt_dur_tag = read_one_meta_file(res_dir, lang)
    res_s = []
    for utt, dur, tag in tqdm.tqdm(utt_dur_tag):
        wav_data, sr = soundfile.read(f"{wav_dir}/{utt}.wav")
        gen_dur = wav_data.shape[0] / sr
        res_s.append([utt, abs(gen_dur - tag), abs(gen_dur - tag) / dur])
    res_m1 = sum(ii[1] for ii in res_s) / len(res_s)
    res_m2 = sum(ii[2] for ii in res_s) / len(res_s)
    sav_lines = [f"{ii[0]} {ii[1]} {ii[2]}\n" for ii in res_s]
    sav_lines.append(f"delta: ADE {res_m1} RDE {res_m2}\n")
    print(f"delta: ADE {res_m1} RDE {res_m2}\n")
    with open(f"{res_dir}/{lang}.dur.time_stretch", "w") as fw:
        fw.write("".join(sav_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir",
        type=str,
    )
    parser.add_argument(
        "--res_dir",
        type=str,
    )
    parser.add_argument(
        "--lang",
        type=str,
    )
    args = parser.parse_args()
    ana_time_stretch(args.wav_dir, args.res_dir, args.lang)


if __name__ == "__main__":
    main()
