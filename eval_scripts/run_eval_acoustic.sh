#!/usr/bin/env python

eval_path=$1
whisper_path=$2
paraformer_path=$3
wavlm_path=$4
eval_mode=$5 # basic | open
lang=$6 # zh | en

export LD_LIBRARY_PATH=""

workdir=$(cd $(dirname $0); pwd)
cd $workdir


for task_type in pitch_shift time_stretch vol dialect; do
  echo "processing ${result}/${task_type}"
  meta_file="../meta/${task_type}/meta_${lang}_${task_type}.csv"

  if [ ! -f "$meta_file" ]; then
    echo "Meta file not found: $meta_file. Skipping."
    continue
  fi

  # wer
  echo $eval_path/$task_type/${task_type}_basic_${lang}/eval_result/wav.wer.${lang}.edited_label
  if [ -e $eval_path/$task_type/${task_type}_basic_${lang}/eval_result/wav.wer.${lang}.edited_label ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_wer_edit.sh $meta_file \
        $eval_path/$task_type/${task_type}_basic_${lang}/tts \
        $lang \
        2 \
        $eval_path/$task_type/${task_type}_basic_${lang}/eval_result \
        $task_type \
        $eval_mode \
        $whisper_path \
        $paraformer_path \
        acoustic

  fi

  # sim
  if [ -e $eval_path/$task_type/${task_type}_basic_${lang}/eval_result/wav_res_ref_text.sim.${lang} ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_sim_edit.sh $meta_file \
        $eval_path/$task_type/${task_type}_basic_${lang}/tts \
        $wavlm_path \
        1 \
        $eval_path/$task_type/${task_type}_basic_${lang}/eval_result \
        ${lang}
  fi
done

# for time_stretch task, calculate Absolute Duration Error (ADE) and Relative Duration Error (RDE)
python pyscripts/ana_time_stretch.py --wav_dir $eval_path/time_stretch/time_stretch_basic_${lang}/tts \
                                     --res_dir $eval_path/time_stretch/time_stretch_basic_${lang}/eval_result \
                                     --lang ${lang}

# for vol changing task, caculate Absolute Amplitude Error (AAE) and Relative Amplitude Error (RAE)
python pyscripts/ana_vol.py --wav_dir $eval_path/vol/vol_basic_${lang}/tts \
                            --res_dir $eval_path/vol/vol_basic_${lang}/eval_result \
                            --lang ${lang}

# uncomment the following code, if you want to calculate the ACC for the dialect conversion task
# you need to set up the url and api key in pyscripts/dialect_api.py

# if [[ ${lang} == "zh"]]; then
#   python pyscripts/dialect_api.py --res_dir $eval_path/dialect/dialect_basic_zh/eval_result \
#                                   --generated_audio_dir $eval_path/dialect/dialect_basic_zh/tts \
#                                   --meta_file ../../meta/dialect/dialect.jsonl
# fi

# uncomment the following code, if you want to calculate the ACC for emotion conversion task
# you need to set up the url and api key in pyscripts/gemini_api.py

# python pyscripts/emo_acc.py --res_dir $eval_path/emotion/emotion_basic_${lang}/eval_result \
#                                 --wav_dir $eval_path/emotion/emotion_basic_${lang}/tts

# dnsmos for speech enhancement task
# sh cal_dnsmos.sh $eval_path/denoise/dns_no_reverb/tts /root/sig_bak_ovr.onnx /root/model_v8.onnx
