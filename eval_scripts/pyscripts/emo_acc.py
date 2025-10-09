import argparse
import multiprocessing
import os
from collections import defaultdict
from tqdm import tqdm
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

def find_files_by_suffix(directory_path: str, suffix: str = ".wav") -> list[list[str]]:
    """
    在指定目录中递归搜索所有以特定后缀结尾的文件。
    Args:
        directory_path: 要搜索的目录路径。
        suffix: 文件后缀名 (例如 '.wav')。

    Returns:
        一个列表，每个元素为 [文件名(不含后缀), 文件的绝对路径]。
    """
    # 确保后缀以点开头，便于匹配
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    
    found_files = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            # 检查文件是否以指定后缀结尾（不区分大小写）
            if filename.lower().endswith(suffix.lower()):
                # 分离文件名和后缀
                name_without_suffix = os.path.splitext(filename)[0]
                # 构建并获取文件的绝对路径
                absolute_path = os.path.abspath(os.path.join(root, filename))
                found_files.append([name_without_suffix, absolute_path])
                
    return found_files

def _reader(task_q, wav_dir, num_proc=20):
    task_s = find_files_by_suffix(wav_dir)

    for utt, wav_name in task_s:
        task_q.put([wav_name, "api", utt])
    for _ in range(num_proc):
        task_q.put(None)


def emo_acc(wav_dir, res_dir, num_proc=20):
    task_q = multiprocessing.Queue(num_proc * 10)
    done_q = multiprocessing.Queue(num_proc * 10)
    workers = [multiprocessing.Process(target=_reader, args=(task_q, wav_dir, num_proc))]
    workers[-1].start()
    for _ in range(num_proc):
        workers.append(multiprocessing.Process(target=_worker, args=(task_q, done_q)))
        workers[-1].start()
    ts = find_files_by_suffix(wav_dir)
    remain, res_d = num_proc, defaultdict(dict)
    while remain:
        info = done_q.get()
        if info is None:
            remain -= 1
            continue
        res_d[info[1]][info[2]] = info[0]
    os.makedirs(res_dir, exist_ok=True)
    for task_name, task_res in res_d.items():
        sav_lines, cnt = [], 0
        for utt, _ in tqdm(ts):
            if utt in task_res:
                one_res = task_res[utt]
                sav_lines.append(f"{utt} {one_res}\n")
                if one_res.split()[-1] in {"happy", "surprised", "joyful", "like"}:
                    cnt += 1
            else:
                sav_lines.append(f"{utt} error error\n")
        sav_lines.append(f"Acc: {cnt/len(ts)}")
        print(f"Acc: {cnt/len(ts)}")
        with open(f"{res_dir}/{task_name}.acc.emotion", "w") as fw:
            fw.write("".join(sav_lines))
    for wk in workers:
        wk.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str)
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--num_proc", type=int, default=20)
    args = parser.parse_args()
    emo_acc(args.wav_dir, args.res_dir, args.num_proc)


if __name__ == "__main__":
    main()
