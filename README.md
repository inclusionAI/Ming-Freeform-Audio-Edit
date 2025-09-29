# README

## Introduction
This repository hosts Ming-Freeform-Audio-Edit, the benchmark test set for evaluating the downstream editing tasks of the Ming-UniAudio model.

This test set covers 7 distinct editing tasks, categorized as follows:

+ Semantic Editing (3 tasks):

  + Free-form Deletion
  + Free-form Insertion
  + Free-form Substitution
+ Acoustic Editing (5 tasks):
  + Time-stretching
  + Pitch Shifting
  + Dialect Conversion
  + Emotion Conversion
  + Volume Conversion

The audio samples are sourced from well-known open-source datasets, including seed-tts eval, LibriTTS, and Gigaspeech.

## Dataset statistics
### Semantic Editing
| Task Types\ # samples \ Language | Zh deletion | Zh insertion | Zh substitution | En deletion | En insertion | En substitution |
| -------------------------------- | ----------: | -----------: | --------------: | ----------: | -----------: | --------------: |
| Index-based                      |         186 |          180 |              36 |         138 |          100 |              67 |
| Content-based                    |          95 |          110 |             289 |          62 |           99 |             189 |
| Total                            |         281 |          290 |             325 |         200 |          199 |             256 |

*Index-based* instruction: specifies an operation on content at positions $i$ to $j$. (e.g. delete the characters or words from index 3 to 12)

*Content-based*: targets specific characters or words for editing. (e.g. insert 'hello' before 'world')
### Acoustic Editing
| Task Types\ # samples \ Language |   Zh |   En |
| -------------------------------- | ---: | ---: |
| Time-stretching                  |   50 |   50 |
| Pitch Shifting                   |   50 |   50 |
| Dialect Conversion               |  250 |  --- |
| Emotion Conversion               |   84 |   72 |
| Volume Conversion                |   50 |   50 |
## Evaluation Metrics
### Semantic Editing
For the deletion, insertion, and substitution tasks, we evaluate the performance using four key metrics:
+ Word Error Rate (WER) of the Edited Region (wer)
+ Word Error Rate (WER) of the Non-edited Region (wer.noedit)
+ Edit Operation Accuracy (acc)
+ Speaker Similarity (sim)

These metrics can be calculated by running the following command:
```bash
# run pip install -r requirements.txt first
bash eval_scripts/semantic/run_eval.sh /path/contains/edited/audios
```
NOTE: the directory passed to the above script should have the structure as follows:
```
.
├── del
│   └── edit_del_basic
│       ├── eval_result
│       ├── hyp.txt
│       ├── input_wavs
│       ├── origin_wavs
│       ├── ref.txt
│       ├── test.jsonl
│       ├── test_parse.jsonl # This is need to run the evaluation script
│       ├── test.yaml
│       └── tts/ # This is the directory contains the edited wavs
```

Examples of test_parse.jsonl:
``` json
{"uid": "00107947-00000092", "input_wav_path": "wavs/00107947-00000092.wav","output_wav_path": "edited_wavs/00107947-00000092.wav", "instruction": "Please recognize the language of this speech and transcribe it. And delete '随着经济的发'.\n", "asr_label": "随着经济的发展食物浪费也随之增长", "asr_text": "随着经济的发展食物浪费也随之增长", "edited_text_label": "展食物浪费也随之增长", "edited_text": "<edit></edit>展食物浪费也随之增长", "origin_speech_url": null,}

{"uid": "00010823-00000019", "input_wav_path": "wavs/00010823-00000019.wav", "output_wav_path": "edited_wavs/00010823-00000019.wav", "instruction": "Please recognize the language of this speech and transcribe it. And delete the characters or words from index 4 to index 10.\n", "asr_label": "我们将为全球城市的可持续发展贡献力量", "asr_text": "我们将为全球城市的可持续发展贡献力量", "edited_text_label": "我们将持续发展贡献力量", "edited_text": "我们将<edit></edit>持续发展贡献力量", "origin_speech_url": null}
```
### Acoustic Editing
For the acoustic editing tasks, we use WER and SPK-SIM as the primary evaluation metrics. These two metrics can be calculated by running the following commands:
```bash
bash eval_scripts/acoustic/cal_wer_sim.sh /path/contains/edited/audios
```

Additionally, for the dialect and emotion conversion tasks, we assess the conversion accuracy by leveraging a large language model (LLM) through API calls.
```bash
# dialect conversion accuracy
python eval_scripts/acoustic/pyscripts/dialect_api.py --output_dir <保存评测结果的根目录> --generated_audio_dir <存放已生成音频文件的目录路径>
# emotion conversion accuracy
# fisrt, run: bash eval_scripts/acoustic/cal_wer_sim.sh /path/contains/edited/audios
python pyscripts/emo_acc.py
```
