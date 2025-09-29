# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/7/29
import argparse
import os

import librosa
import numpy as np
import torch
import torchaudio
import tqdm
from ecapa_tdnn import ECAPA_TDNN_SMALL


def read_utt2vec(ts_dir, lan, prompt_dir, device):
    with open(f"{ts_dir}/{lan}/meta.lst", "r") as fr:
        zh_utt_vec = [
            ii.split("|")[0] for ii in fr.read().splitlines() if ii.split("|")[0] != "file_name"
        ]
    res = []
    for utt in zh_utt_vec:
        for prefix in ["time_stretch_", "pitch_shift_", "vol_", "emotion_", "dialect_"]:
            if utt.startswith(prefix):
                res.append([utt, f"../../wavs/{utt[len(prefix):]}.wav"])
                break
        else:
            res.append([utt, f"../../wavs/{utt}.wav"])
    return res


def sv_sim(task_file, ts_dir, mdl_dir, prompt_dir, verbose_dir, rank=0, num_proc=1, dialect=False):
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024, feat_type="wavlm_large", config_path=None, f_ext_dir=mdl_dir
    )
    state_dict = torch.load(
        f"{mdl_dir}/wavlm_large_finetune.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(state_dict["model"], strict=False)
    device = f"cuda:{rank}"
    model.to(device).eval()

    with open(task_file, "r") as fr:
        task_lines = fr.read().splitlines()
    os.makedirs(verbose_dir, exist_ok=True)

    zh_utt_vec = read_utt2vec(ts_dir, "zh", prompt_dir, device)
    if not dialect:
        en_utt_vec = read_utt2vec(ts_dir, "en", prompt_dir, device)

    if rank == 0:
        task_iter = tqdm.tqdm(task_lines[rank::num_proc])
    else:
        task_iter = task_lines[rank::num_proc]
    for task_line in task_iter:
        mdl_n, _, wav_dir = task_line.split()
        if not dialect:
            vec_list = [("zh", zh_utt_vec), ("en", en_utt_vec)]
        else:
            vec_list = [("zh", zh_utt_vec)]
        for lan, utt_vec in vec_list:
            sav_lines, sim_s = [], []
            for utt, vec in utt_vec:
                if isinstance(vec, str):
                    p_wav, p_sr = torchaudio.load(vec, backend="soundfile")
                    if p_sr != 16000:
                        p_wav_numpy = p_wav.numpy()
                        resampled_p_wav_numpy = librosa.resample(
                            p_wav_numpy, orig_sr=p_sr, target_sr=16000
                        )
                        p_wav = torch.from_numpy(resampled_p_wav_numpy)
                        p_sr = 16000
                    assert p_sr == 16000
                    vec = model(p_wav.to(device))
                wav, sr = torchaudio.load(f"{wav_dir}/{utt}.wav", backend="soundfile")
                if sr != 16000:
                    wav_numpy = wav.numpy()
                    resampled_wav_numpy = librosa.resample(wav_numpy, orig_sr=sr, target_sr=16000)
                    wav = torch.from_numpy(resampled_wav_numpy)
                    sr = 16000
                assert sr == 16000
                gen_vec = model(wav.to(device))
                sim = torch.nn.functional.cosine_similarity(vec, gen_vec).item()
                sav_lines.append(f"{utt} {sim}\n")
                sim_s.append(sim)
            sim_s = np.array(sim_s)
            sav_lines.append(f"mdl_n,ASV,std\n")
            sav_lines.append(f"{mdl_n},{sim_s.mean()},{sim_s.std()}\n")
            with open(f"{verbose_dir}/{mdl_n}_{lan}.log", "w") as fw:
                fw.write("".join(sav_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, default="config/task.lst")
    parser.add_argument("--verbose_dir", type=str, default="logs_sim")
    parser.add_argument(
        "--ts_dir",
        type=str,
        default="resource/seedtts_testset",
    )
    parser.add_argument(
        "--mdl_dir",
        type=str,
        default="Models/WavLM-large",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="resource/seed_sim",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--dialect", type=bool, default=False)
    args = parser.parse_args()
    sv_sim(
        args.task_file,
        args.ts_dir,
        args.mdl_dir,
        args.prompt_dir,
        args.verbose_dir,
        args.rank,
        args.num_proc,
        args.dialect,
    )


if __name__ == "__main__":
    main()
