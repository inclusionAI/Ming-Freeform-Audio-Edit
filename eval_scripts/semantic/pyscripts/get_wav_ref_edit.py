import json
import os
import sys

from tqdm import tqdm

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_label_edited = sys.argv[3]

data = []
with open(metalst, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


with open(wav_label_edited, "w") as f:
    for line in tqdm(data, desc="generating"):
        uid, text_label, edited_text, prompt_wav = (
            line["uid"],
            line["edited_text_label"],
            line["edited_text"],
            line["input_wav_path"],
        )
        if not os.path.exists(os.path.join(wav_dir, uid + ".wav")):
            continue

        out_line = "|".join(
            [os.path.join(wav_dir, uid + ".wav"), prompt_wav, text_label, edited_text]
        )

        f.write(out_line + "\n")
