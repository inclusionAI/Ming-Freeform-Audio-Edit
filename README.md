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
#### full version
| Task Types\ # samples \ Language | Zh deletion | Zh insertion | Zh substitution | En deletion | En insertion | En substitution |
| -------------------------------- | ----------: | -----------: | --------------: | ----------: | -----------: | --------------: |
| Index-based                      |         186 |          180 |              36 |         138 |          100 |              67 |
| Content-based                    |          95 |          110 |             289 |          62 |           99 |             189 |
| Total                            |         281 |          290 |             325 |         200 |          199 |             256 |

#### basic version
| Task Types\ # samples \ Language | Zh deletion | Zh insertion | Zh substitution | En deletion | En insertion | En substitution |
| -------------------------------- | ----------: | -----------: | --------------: | ----------: | -----------: | --------------: |
| Index-based                      |          92 |           65 |              29 |          47 |           79 |              29 |
| Content-based                    |          78 |          105 |             130 |         133 |           81 |             150 |
| Total                            |         170 |          170 |             159 |         180 |          160 |             179 |

*Index-based* instruction: specifies an operation on content at positions *i* to *j*. (e.g. delete the characters or words from index 3 to 12)

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
### Environment Preparation
```bash
git clone https://github.com/inclusionAI/Ming-Freeform-Audio-Edit.git
cd Ming-Freeform-Audio-Edit
pip install -r requirements.txt
```
**Note**: Please download the audio and meta files from [HuggingFace](https://huggingface.co/datasets/inclusionAI/Ming-Freeform-Audio-Edit-Benchmark/tree/main) or [ModelScope](https://modelscope.cn/datasets/inclusionAI/Ming-Freeform-Audio-Edit-Benchmark/files) and put the `wavs` and `meta` directories under `Ming-Freeform-Audio-Edit`
### Semantic Editing
For the deletion, insertion, and substitution tasks, we evaluate the performance using four key metrics:
+ Word Error Rate (WER) of the Edited Region (wer)
+ Word Error Rate (WER) of the Non-edited Region (wer.noedit)
+ Edit Operation Accuracy (acc)
+ Speaker Similarity (sim)

1. If you have organized the directories contain edited waveforms like below:
   ```
   eval_path
    |
    ├── del
    │   └── edit_del_basic
    │       └── tts/ # This is the actual directory contains the edited wavs
    ├── ins
    │   └── edit_ins_basic
    │       └── tts/ # This is the actual directory contains the edited wavs
    ├── sub
      └── edit_sub_basic
          └── tts/ # This is the actual directory contains the edited wavs
   ```
   Then you can run the following command to get those metrics:
   ```bash
   cd Ming-Freeform-Audio-Edit/eval_scripts
   bash run_eval_semantic.sh eval_path \
                             whisper_path \
                             paraformer_path \
                             wavlm_path \
                             eval_mode \
                             lang
   ```
   Here is a brief description of the parameters for the script above:
     + `eval_path`: The top-level directory containing subdirectories for each editing task
     + `whisper_path`:Path to the Whisper model, which is used to calculate WER for English audio. You can download it from [here](https://huggingface.co/openai/whisper-large-v3).
     + `paraformer_path`:Path to the Paraformer model, which is used to calculate WER for Chinese audio. You can download it from [here](https://huggingface.co/funasr/paraformer-zh).
     + `wavlm_path`: Path to the WavLM model, which is used to calculate speaker similarity. You can download it from [here](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).
     + `eval_mode`: Used to specify which version of the evaluation set to use. Choose between `basic` and `open`
     + `lang`: supported language, choose between `zh` and `en`

2. If your directory for the edited audio is not organized in the format described above, you can run the following commands.
   ```bash
   cd eval_scripts
   # get wer, wer.noedit
   bash cal_wer_edit.sh meta_file \
        wav_dir \
        lang \
        num_jobs \
        res_dir \
        task_type \
        eval_mode \
        whisper_path \
        paraformer_path \
        edit_cat # use `semantic` here
    # get sim
    bash cal_sim_edit.sh meta_file \
         wav_dir \
         wavlm_path \
         num_jobs \
         res_dir \
         lang
   ```
   Here is a brief description of the parameters for the script above:
   + `meta_file`: The absolute path to the meta file for the corresponding task (e.g., `meta_en_deletion_basic.csv` or `meta_en_deletion.csv`).
   + `wav_dir`: The directory containing the edited audio files (the WAV files should be located directly in this directory).
   + `lang`: `zh` or `en`
   + `num_jobs`: number of process.
   + `res_dir`: The directory to save the metric results.
   + `task_type`: `del`, `ins` or `sub`
   + `eval_mode`: The same as the above.
   + `whisper_path`: The same as the above
   + `paraformer_path`: The same as the above
   + `edit_cat`: `semantic` or `acoustic`


### Acoustic Editing
For the acoustic editing tasks, we use WER and SPK-SIM as the primary evaluation metrics. 

1. If the directory for the edited audio is structured, you can run the following command.
   ```bash
   cd Ming-Freeform-Audio-Edit/eval_scripts
   bash run_eval_acoustic.sh eval_path \
                            whisper_path \
                            paraformer_path \
                            wavlm_path \
                            eval_mode \
                            lang
   ```
2. Otherwise, you can run commands similar to the one for the semantic tasks, with the `edit_cat` parameter set to `acoustic`.

Additionally, for the dialect and emotion conversion tasks, we assess the conversion accuracy by leveraging a large language model (LLM) through API calls, refer to `eval_scripts/run_eval_acoustic.sh` for more details.
