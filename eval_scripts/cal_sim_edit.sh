#!/usr/bin/env python

meta_lst=$1
wav_dir=$2
checkpoint_path=$3
jobs=$4
result_dir=$5
lang=$6

export LD_LIBRARY_PATH=""

[ -d "$result_dir" ] || mkdir -p "$result_dir"

wav_wav_text=$result_dir/wav_res_ref_text
score_file=$result_dir/wav_res_ref_text.sim.${lang}

python3 pyscripts/get_wav_ref_edit.py $meta_lst $wav_dir $wav_wav_text

if command -v nvidia-smi &>/dev/null; then
    NUM_PROCESSES=$(nvidia-smi -L | wc -l)
else
    NUM_PROCESSES=$(ppu-smi -L | wc -l)
fi


workdir=$(cd $(dirname $0); cd ../; pwd)

cd $workdir/thirdparty/UniSpeech/downstreams/speaker_verification/

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
        python3 verification_pair_list_v2.py $thread_dir/thread-0$rank.lst \
            --model_name wavlm_large \
            --checkpoint $checkpoint_path \
            --scores $out_dir/thread-0$rank.sim.out \
            --wav1_start_sr 0 \
            --wav2_start_sr 0 \
            --wav1_end_sr -1 \
            --wav2_end_sr -1 \
            --device cuda:$gpu_id &
    done
fi
wait

rm $wav_wav_text
rm -f $out_dir/merge.out

cat $out_dir/thread-0*.sim.out | grep -v "avg score" >>  $out_dir/merge.out
python3 average.py $out_dir/merge.out $score_file
