eval_path=$1

workdir=$(cd $(dirname $0); pwd)
cd $workdir

for task_type in ins del sub; do
  echo "processing ${result}/${task_type}"

  # # wer
  echo $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav.wer.zh.edited_label.edited_acc
  if [ -e $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav.wer.zh.edited_label.edited_acc ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_wer_edit.sh $eval_path/$task_type/edit_${task_type}_basic/test_parse.jsonl \
        $eval_path/$task_type/edit_${task_type}_basic/tts \
        zh \
        2 \
        $eval_path/$task_type/edit_${task_type}_basic/eval_result \
        $task_type
  fi

  #sim
  if [ -e $eval_path/$task_type/edit_${task_type}_basic/eval_result/wav_res_ref_text.sim.zh ]; then
    echo "$eval_path/$task_type exist"
  else
    sh  cal_sim_edit.sh $eval_path/$task_type/edit_${task_type}_basic/test_parse.jsonl \
        $eval_path/$task_type/edit_${task_type}_basic/tts \
        /input/linyi/Models/WavLM-large/wavlm_large_finetune.pth \
        1 \
        $eval_path/$task_type/edit_${task_type}_basic/eval_result \
        zh
  fi
done