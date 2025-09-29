# set -x

meta_lst=$1
wav_dir=$2
lang=$3
jobs=$4
result_dir=$5
task=$6
eval_mode=$7

mkdir $result_dir

wav_wav_text=$result_dir/wav_res_ref_text
audio_score_file=$result_dir/wav.wer.${lang}
text_score_file=$result_dir/text.wer.${lang}


python3 pyscripts/get_wav_ref_edit.py $meta_lst $wav_dir $wav_wav_text


if command -v nvidia-smi &>/dev/null; then
    NUM_PROCESSES=$(nvidia-smi -L | wc -l)
else
    NUM_PROCESSES=$(ppu-smi -L | wc -l)
fi


timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp/
mkdir $thread_dir
num_job=$jobs
echo $num_job
num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`
sudo split -l $num_per_thread --additional-suffix=.lst -d $wav_wav_text $thread_dir/thread-
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir $out_dir

num_job_minus_1=`expr $num_job - 1`
if [ ${num_job_minus_1} -ge 0 ];then
	for rank in $(seq 0 $((num_job - 1))); do
        gpu_id=$((rank % ${NUM_PROCESSES}))
        echo $gpu_id
		sub_score_file=$out_dir/thread-0$rank.wer.out
		CUDA_VISIBLE_DEVICES=${gpu_id} python3 pyscripts/run_wer.py $thread_dir/thread-0$rank.lst $sub_score_file $lang "asr" &
	done
fi
wait

rm -f $out_dir/merge.out

cat $out_dir/thread-0*.wer.out.edited_label >>  $out_dir/merge.out.edited_label
python3 pyscripts/average_wer.py $out_dir/merge.out.edited_label $audio_score_file.edited_label

cat $out_dir/thread-0*.wer.out.edited >>  $out_dir/merge.out.edited
python3 pyscripts/average_wer.py $out_dir/merge.out.edited $audio_score_file.edited

# cal wer for edited_text and text_label
python3 pyscripts/run_wer.py $wav_wav_text $text_score_file.tmp $lang "wer"
python3 pyscripts/average_wer.py $text_score_file.tmp $text_score_file
rm -f $text_score_file.tmp
rm $wav_wav_text


if [[ "$eval_mode" == "open" || "$lang" == "en" ]]; then
    # cal acc and wer wav.wer.zh.edited_label, 音频跟随能力
    python3 pyscripts/get_ref_hyp.py $audio_score_file.edited_label $audio_score_file.edited_label_x1 $audio_score_file.edited_label_x2
    python3 pyscripts/eval_wer.py $audio_score_file.edited_label_x1 $audio_score_file.edited_label_x2 --tochar --verbose > $audio_score_file.edited_label.new # 修复wer指标
    python3 pyscripts/get_acc.py $meta_lst $task --wer_file $audio_score_file.edited_label.new --ref_file $audio_score_file.edited_label_ref --hyp_file $audio_score_file.edited_label_hyp --eval_mode open --lang $lang > $audio_score_file.edited_label.edited_acc
    python3 pyscripts/eval_wer.py $audio_score_file.edited_label_ref $audio_score_file.edited_label_hyp --tochar --verbose > $audio_score_file.edited_label.noedit_wer
    rm -rf $audio_score_file.edited_label_*
else
    case "$task" in
        sub|ins)
            # cal acc and wer wav.wer.zh.edited_label, 音频跟随能力
            python3 pyscripts/get_ref_hyp.py $audio_score_file.edited_label $audio_score_file.edited_label_x1 $audio_score_file.edited_label_x2
            python3 pyscripts/eval_wer.py $audio_score_file.edited_label_x1 $audio_score_file.edited_label_x2 --tochar --verbose > $audio_score_file.edited_label_wer
            python3 pyscripts/get_acc.py $meta_lst $task --wer_file $audio_score_file.edited_label_wer --ref_file $audio_score_file.edited_label_ref --hyp_file $audio_score_file.edited_label_hyp --eval_mode open > $audio_score_file.edited_label.edited_acc
            python3 pyscripts/eval_wer.py $audio_score_file.edited_label_ref $audio_score_file.edited_label_hyp --tochar --verbose > $audio_score_file.edited_label.noedit_wer
            rm -rf $audio_score_file.edited_label_*
            ;;
        del)
            # cal acc and wer wav.wer.zh.edited_label, 音频跟随能力
            python3 pyscripts/get_ref_hyp.py $audio_score_file.edited_label $audio_score_file.edited_label_x1 $audio_score_file.edited_label_x2
            sed -i 's/ /\t/; s/ //g; s/\t/ /' $audio_score_file.edited_label_x2
            python3 pyscripts/get_acc.py $meta_lst $task --ref_file $audio_score_file.edited_label_ref --hyp_file $audio_score_file.edited_label_hyp --wav_asr_text $audio_score_file.edited_label_x2 > $audio_score_file.edited_label.edited_acc
            python3 pyscripts/eval_wer.py $audio_score_file.edited_label_ref $audio_score_file.edited_label_hyp --tochar --verbose > $audio_score_file.edited_label.noedit_wer
            rm -rf $audio_score_file.edited_label_*
            ;;
        *)
    esac
fi
