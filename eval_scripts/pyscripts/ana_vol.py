import argparse
import os

import numpy as np
import soundfile
import tqdm


def read_one_meta_file(res_dir, lang):
    if os.path.exists(f"{res_dir}/tag.vol.vol"):
        with open(f"{res_dir}/tag.vol.vol", "r") as fr:
            utt_vol_tag = [ii.split() for ii in fr.read().splitlines()]
        utt_vol_tag = [[ii[0], float(ii[1]), float(ii[2])] for ii in utt_vol_tag]
    else:
        with open(f"../meta/vol/meta_{lang}_vol.csv", "r") as fr:
            meta_lines = fr.read().splitlines()
        sav_lines, utt_vol_tag = [], []
        for meta_line in meta_lines:
            meta_line = meta_line.split("|")
            assert len(meta_line) == 5
            if not (
                meta_line[4].startswith("vol(") or meta_line[2].startswith("adjusts the volume to")
            ):
                continue
            utt = meta_line[0]
            if meta_line[4].startswith("vol("):
                prompt_args = float(meta_line[4].split("(")[1])
            else:
                prompt_args = float(meta_line[2].rsplit(maxsplit=1)[1])
            pro_utt = utt[len("vol_") :] if utt.startswith("vol_") else utt
            wav_data, sr = soundfile.read(f"../wavs/{pro_utt}.wav")
            vol = np.abs(wav_data).mean()
            tag = wav_data * prompt_args
            tag = np.abs(tag).mean()
            sav_lines.append(f"{utt} {vol} {tag}\n")
            utt_vol_tag.append([utt, vol, tag])
        with open(f"{res_dir}/tag.vol.vol", "w") as fw:
            fw.write("".join(sav_lines))
    return utt_vol_tag


def ana_vol(wav_dir, res_dir, lang):
    os.makedirs(res_dir, exist_ok=True)
    utt_vol_tag = read_one_meta_file(res_dir, lang)

    res_s = []
    for utt, vol, tag in tqdm.tqdm(utt_vol_tag):
        wav_data, sr = soundfile.read(f"{wav_dir}/{utt}.wav")
        gen_vol = np.abs(wav_data).mean()
        res_s.append([utt, abs(gen_vol - tag), abs(gen_vol - tag) / vol])
    res_m1 = sum(ii[1] for ii in res_s) / len(res_s)
    res_m2 = sum(ii[2] for ii in res_s) / len(res_s)
    sav_lines = [f"{ii[0]} {ii[1]} {ii[2]}\n" for ii in res_s]
    sav_lines.append(f"delta: AAE {res_m1} RAE {res_m2}\n")
    print(f"delta: AAE {res_m1} RAE {res_m2}\n")
    with open(f"{res_dir}/{lang}.vol.vol", "w") as fw:
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
    ana_vol(args.wav_dir, args.res_dir, args.lang)


if __name__ == "__main__":
    main()
