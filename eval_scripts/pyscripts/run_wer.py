import os
import string
import sys

import scipy
import soundfile as sf
import zhconv
from funasr import AutoModel
from jiwer import compute_measures
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from zhon.hanzi import punctuation

punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3]  # zh or en
task = sys.argv[4]  # asr or wer only
whisper_path = sys.argv[5]
paraformer_path = sys.argv[6]
device = "cuda:0"


def load_en_model(model_id):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model


def load_zh_model(model):
    model = AutoModel(
        model=model,
    )
    return model


def process_one(hypo, truth):
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

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path, whisper_path, paraformer_path):
    if lang == "en":
        processor, model = load_en_model(model_id=whisper_path)
    elif lang == "zh":
        model = load_zh_model(model=paraformer_path)

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split("|")) == 4:
            wav_res_path, _, text_ref, edited_text = line.split("|")
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref, edited_text))
    fout = open(f"{res_path}.edited_label", "w")

    # wer for audio and edited_text_label
    for wav_res_path, text_ref, edited_text in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_res_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif lang == "zh":
            try:
                res = model.generate(input=wav_res_path, batch_size_s=300)
                transcription = res[0]["text"]
                transcription = zhconv.convert(transcription, "zh-cn")
            except:
                transcription = ""

        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()


def run_wer(wav_res_text_path, res_path):
    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        wav_res_path, _, text_ref, edited_text = line.split("|")

        params.append((wav_res_path, text_ref, edited_text))
    fout = open(res_path, "w")
    for line in open(wav_res_text_path).readlines():
        if len(line.split("|")) == 4:
            _, _, text_ref, edited_text = line.split("|")
        else:
            raise NotImplementedError

        if lang == "en":
            pass
        elif lang == "zh":
            edited_text = edited_text.strip()
            edited_text = " ".join(edited_text)
        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(edited_text, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()


if task == "asr":
    run_asr(wav_res_text_path, res_path, whisper_path, paraformer_path)
elif task == "wer":
    run_wer(wav_res_text_path, res_path)
else:
    raise NotImplementedError
