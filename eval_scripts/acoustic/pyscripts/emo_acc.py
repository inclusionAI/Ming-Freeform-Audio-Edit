# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/9/16
import argparse
import multiprocessing
import os
from collections import defaultdict

from gemini_api import GeminiApi


def _worker(task_q, done_q):
    while True:
        info = task_q.get()
        if info is None:
            break
        if not os.path.exists(info[0]):
            info[0] = "error error"
        else:
            try:
                result = GeminiApi.api_audio_emotion(info[0])
                if len(result) == 0:
                    result = "error error"
            except Exception as e:
                print(e)
                result = "error error"
            info[0] = result
        done_q.put(info)
    done_q.put(None)


def read_one_ts(ts_dir, lang):
    with open(f"{ts_dir}/{lang}/meta.lst", "r") as fr:
        meta_lines = fr.read().splitlines()
    res_s = []
    for meta_line in meta_lines:
        meta_line = meta_line.split("|")
        assert len(meta_line) > 0
        res_s.append([f"{meta_line[0]}", f"{meta_line[0]}.wav"])
    return res_s


def _reader(task_q, task_file, ts_dir, num_proc=20):
    task_s = read_one_ts(ts_dir, "zh") + read_one_ts(ts_dir, "en")
    with open(task_file, "r") as fr:
        task_lines = fr.read().splitlines()
    for task_line in task_lines:
        task_name, _, wav_dir = task_line.split()
        for utt, wav_name in task_s:
            task_q.put([f"{wav_dir}/{wav_name}", task_name, utt])
    for _ in range(num_proc):
        task_q.put(None)


def emo_acc(task_file, ts_dir, verbose_dir, num_proc=20):
    task_q = multiprocessing.Queue(num_proc * 10)
    done_q = multiprocessing.Queue(num_proc * 10)
    workers = [multiprocessing.Process(target=_reader, args=(task_q, task_file, ts_dir, num_proc))]
    workers[-1].start()
    for _ in range(num_proc):
        workers.append(multiprocessing.Process(target=_worker, args=(task_q, done_q)))
        workers[-1].start()
    zh_ts = read_one_ts(ts_dir, "zh")
    en_ts = read_one_ts(ts_dir, "en")
    remain, res_d = num_proc, defaultdict(dict)
    while remain:
        info = done_q.get()
        if info is None:
            remain -= 1
            continue
        res_d[info[1]][info[2]] = info[0]
    os.makedirs(verbose_dir, exist_ok=True)
    for task_name, task_res in res_d.items():
        for lang, lang_ts in [["zh", zh_ts], ["en", en_ts]]:
            sav_lines, cnt = [], 0
            for utt, _ in lang_ts:
                if utt in task_res:
                    one_res = task_res[utt]
                    sav_lines.append(f"{utt} {one_res}\n")
                    if one_res.split()[-1] in {"happy", "surprised", "joyful", "like"}:
                        cnt += 1
                else:
                    sav_lines.append(f"{utt} error error\n")
            sav_lines.append(f"Acc: {cnt/len(lang_ts)}")
            with open(f"{verbose_dir}/{task_name}.{lang}.acc.emotion", "w") as fw:
                fw.write("".join(sav_lines))
    for wk in workers:
        wk.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, default="config/task.lst")
    parser.add_argument("--verbose_dir", type=str, default="logs_acc_emotion")
    parser.add_argument("--ts_dir", type=str, default="emotion")
    parser.add_argument("--num_proc", type=int, default=20)
    args = parser.parse_args()
    emo_acc(args.task_file, args.ts_dir, args.verbose_dir, args.num_proc)


if __name__ == "__main__":
    main()
