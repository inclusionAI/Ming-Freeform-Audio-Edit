#!/bin/bash
nj=2  # Number of parallel CPU/GPU jobs for speedup
python=python3

if [ "$#" -ne 3 ]; then
    echo "错误：需要提供三个参数。"
    echo "用法: $0 <推理结果所在目录> <primary_model路径> <p808_model路径>"
    exit 1
fi

# Whether to use GPU for inference
gpu_inference=true
if ${gpu_inference}; then
    _device="cuda"
else
    _device="cpu"
fi

# you should give the ref and inf dirs
inf_dir=$1
primary_model=$2
p808_model=$3

ext=wav

conda activate edit
export LD_LIBRARY_PATH=""

workdir=$(cd $(dirname $0); pwd)
cd $workdir

find "$(realpath "$inf_dir")" -name "*.$ext" | \
    awk -F'/' -v ext="$ext" '{utt_id=$NF; sub("\\."ext"$", "", utt_id); print utt_id" "$0}' | \
    LC_ALL=C sort -u > enhanced.scp

inf_scp=enhanced.scp
output_prefix=$(realpath "$inf_dir")/../eval_result

pids=() # initialize pids
for idx in $(seq ${nj}); do
(

    # Run each parallel job on a different GPU (if $gpu_inference = true)
    CUDA_VISIBLE_DEVICES=$((${idx} - 1 )) ${python} pyscripts/calculate_dnsmos_metrics.py \
        --inf_scp "${inf_scp}" \
        --output_dir "${output_prefix}"/scoring_dnsmos \
        --device ${_device} \
        --nsplits ${nj} \
        --job ${idx} \
        --convert_to_torch ${gpu_inference} \
        --primary_model ${primary_model} \
        --p808_model ${p808_model}

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs were failed." && false
echo "Finished"

if [ ${nj} -gt 1 ]; then
    for i in $(seq ${nj}); do
        cat "${output_prefix}"/scoring_dnsmos/DNSMOS_OVRL.${i}.scp
    done > "${output_prefix}"/scoring_dnsmos/DNSMOS_OVRL.scp

    for i in $(seq ${nj}); do
        cat "${output_prefix}"/scoring_dnsmos/DNSMOS_SIG.${i}.scp
    done > "${output_prefix}"/scoring_dnsmos/DNSMOS_SIG.scp

    for i in $(seq ${nj}); do
        cat "${output_prefix}"/scoring_dnsmos/DNSMOS_BAK.${i}.scp
    done > "${output_prefix}"/scoring_dnsmos/DNSMOS_BAK.scp

    python - <<EOF
import json
import numpy as np
import os

metrics = ["DNSMOS_OVRL", "DNSMOS_SIG", "DNSMOS_BAK"]
# 用于存储每个指标最终平均值的字典
mean_results = {}
# 遍历每个指标文件
for metric in metrics:
    scores = []
    path = f"${output_prefix}/scoring_dnsmos/{metric}.scp"
    
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # 取行末作为分数
                try:
                    scores.append(float(parts[-1]))
                except ValueError:
                    # 如果分数转换失败，跳过该行
                    print(f"警告: 无法解析分数，跳过此行: {line.strip()}")
    except FileNotFoundError:
        print(f"警告: 指标文件未找到，已跳过: {path}")

    # 计算该指标的平均分
    # 使用 np.nanmean 可以在分数列表为空时返回 nan，避免错误
    if scores:
        mean_score = np.mean(scores)
    else:
        mean_score = float("nan") # 或者可以设为 0 或 None

    # 将指标和其平均分存入结果字典
    mean_results[metric] = mean_score

# 定义输出 JSON 文件的路径
output_json_path = f"${output_prefix}/scoring_dnsmos/RESULTS.jsonl"


existing_data = []

# 将最新的结果添加到现有数据中
existing_data.append(mean_results)
with open(output_json_path, "w") as f:
    for entry in existing_data:
        f.write(json.dumps(entry) + "\n")

print(f"所有指标的平均分已成功保存到: {output_json_path}")
EOF
fi

rm -f ${inf_scp}