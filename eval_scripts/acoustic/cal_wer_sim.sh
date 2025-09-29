eval_path=$1

mkdir -p "${eval_path}/eval_result"

workdir=$(cd $(dirname $0); pwd)
echo $workdir
cd $workdir

rm -rf api_wavs
mkdir api_wavs
cd api_wavs
for edit_task in time_stretch pitch_shift vol emotion;do
  for lang in zh en;do
    ln -s ${eval_path}/${edit_task}/${edit_task}_basic_${lang}/tts/* .
  done
done
cd ..
mkdir -p config
echo "api xxx api_wavs" > config/task.lst

for edit_task in time_stretch pitch_shift vol emotion;do
  if [ ! -d ${edit_task} ];then
    mkdir -p ${edit_task}/zh ${edit_task}/en
    cp ../../meta/${edit_task}/meta_zh_${edit_task}.csv ${edit_task}/zh/meta.lst
    cp ../../meta/${edit_task}/meta_en_${edit_task}.csv ${edit_task}/en/meta.lst
  fi
  python pyscripts/asr_wer.py --batch_size=1 --ts_dir=${edit_task} --verbose_dir=logs_wer_${edit_task}
  python pyscripts/sv_sim.py --ts_dir=${edit_task} --verbose_dir=logs_sim_${edit_task}
  cp logs_wer_${edit_task}/api_zh.log ${eval_path}/eval_result/${edit_task}.zh.wer
  cp logs_wer_${edit_task}/api_en.log ${eval_path}/eval_result/${edit_task}.en.wer
  cp logs_sim_${edit_task}/api_zh.log ${eval_path}/eval_result/${edit_task}.zh.sim
  cp logs_sim_${edit_task}/api_en.log ${eval_path}/eval_result/${edit_task}.en.sim
done
python pyscripts/ana_time_stretch.py
cp logs_time_stretch_dur/api.zh.dur.time_stretch ${eval_path}/eval_result/time_stretch.zh.dur
cp logs_time_stretch_dur/api.en.dur.time_stretch ${eval_path}/eval_result/time_stretch.en.dur
# 音量
python pyscripts/ana_vol.py

# 方言
mkdir -p dialect/zh
cp ../../meta/dialect/meta_zh_dialect.csv dialect/zh/meta.lst
python pyscripts/asr_wer.py --batch_size=1 --ts_dir=dialect --verbose_dir=logs_wer_dialect --dialect True
python pyscripts/sv_sim.py --ts_dir=dialect --verbose_dir=logs_sim_dialect --dialect True