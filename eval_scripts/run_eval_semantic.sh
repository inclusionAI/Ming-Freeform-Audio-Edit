#!/usr/bin/env python

eval_path=$1
whisper_path=$2
paraformer_path=$3
wavlm_path=$4
eval_mode=$5 # basic | open
lang=$6 # zh | en

export LD_LIBRARY_PATH=""

declare -A task_2_meta=(
    ["del"]="deletion"
    ["ins"]="insertion"
    ["sub"]="substitution"
)

workdir=$(cd $(dirname $0); pwd)
cd $workdir


for task_type in ins del sub; do
  echo "processing ${result}/${task_type}"
  if [ $eval_mode == 'basic' ]; then
    meta_file="../meta/${task_type}/meta_${lang}_${task_2_meta[${task_type}]}_basic.csv"
  else
    meta_file="../meta/${task_type}/meta_${lang}_${task_2_meta[${task_type}]}.csv"
  fi

  if [ ! -f "$meta_file" ]; then
    echo "Meta file not found: $meta_file. Skipping."
    continue
  fi

  # wer
  echo $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav.wer.zh.edited_label.edited_acc
  if [ -e $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav.wer.zh.edited_label.edited_acc ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_wer_edit.sh $meta_file \
        $eval_path/$task_type/edit_${task_type}_basic/tts \
        $lang \
        2 \
        $eval_path/$task_type/edit_${task_type}_basic/eval_result \
        $task_type \
        $eval_mode \
        $whisper_path \
        $paraformer_path \
        semantic

  fi

  # sim
  if [ -e $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav_res_ref_text.sim.${lang} ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_sim_edit.sh $meta_file \
        $eval_path/$task_type/edit_${task_type}_basic/tts \
        $wavlm_path \
        1 \
        $eval_path/$task_type/edit_${task_type}_basic/eval_result \
        ${lang}
  fi
done