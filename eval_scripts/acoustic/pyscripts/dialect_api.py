import base64
import json
import os
import os.path
import re
import time

import requests
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

API_URL = "https://matrixllm.alipay.com/v1/chat/completions"
API_HEADERS = {
    "Content-Type": "application/json",
    # 你需要在环境变量中设置 Authorization
    "Authorization": os.environ.get("Authorization", ""),
}
# 定义 lang 字段到 API 预期关键词的映射
LANG_TO_DIALECT_KEYWORD = {
    "zh_chengdu": "四川话",
    "zh_dongbei": "东北话",
    "zh_guangxi": "广西话",
}


def audio_to_base64_string(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        return base64.b64encode(audio_bytes).decode("utf-8")


def extract_clean_text_regex(line):
    """
    使用正则表达式从带有标签的行中提取纯净文本。
    """
    # 使用 re.sub() 函数，将所有匹配 '<...>' 的部分替换为空字符串 ''
    clean_text = re.sub(r"<[^>]*>", "", line)

    return clean_text.strip()


class Dialect_Evaluation:
    def __init__(self, output_dir: str = "/root/outputs"):
        """
        Args:
            output_dir (str, optional): 保存结果的根目录。
        """
        self.accelerator = Accelerator()
        logger.info(f"使用 Accelerator 在设备 {self.accelerator.device} 上加载模型...")

        self.output_dir = f"{output_dir}/eval_result"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _call_dialect_api(self, audio_path: str) -> str:
        """
        调用方言识别 API 并返回模型的文本响应。
        """
        try:
            base64_audio = audio_to_base64_string(audio_path)
            payload = {
                "stream": True,
                "model": "gemini-2.5-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请识别这段音频中的中文方言。请回复方言类型：东北话、四川话、广西话。\n如果不是以上方言类型，请回答“其他”。",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"format": "audio/wav", "data": base64_audio},
                            },
                        ],
                    }
                ],
            }
            response = requests.post(
                API_URL, headers=API_HEADERS, json=payload, stream=True, timeout=60
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        json_str = decoded_line[5:].strip()
                        if json_str and json_str != "[DONE]":
                            data = json.loads(json_str)
                            full_response += (
                                data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            )
            return full_response.strip()
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return f"API_ERROR: {e}"

    def evaluate_from_folder(self, generated_audio_dir: str):
        """
        直接从一个包含已生成音频的文件夹计算评测指标，跳过模型推理步骤。
        (已修改为流式写入，防止数据丢失)

        Args:
            testset_name (str): 要评测的测试集名称。
            generated_audio_dir (str): 存放已生成 .wav 文件的目录路径。
                                       文件名应与 uid 匹配 (e.g., "uid123.wav")。
        """
        self.accelerator.wait_for_everyone()
        start_time = time.time()

        # 创建本次评测的独立输出目录
        eval_output_dir = f"{self.output_dir}/dialect_api"
        if self.accelerator.is_main_process:
            os.makedirs(eval_output_dir, exist_ok=True)

        logger.info(f"音频文件夹： {generated_audio_dir}")

        # 加载数据集元数据
        data = []
        with open("meta/dialect/dialect.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        metrics = "api_judge"

        # === 新增：为每个进程创建一个临时的结果文件 ===
        # 使用 os.path.join 来确保路径的正确性
        temp_result_file_path = os.path.join(
            eval_output_dir, f"temp_results_{self.accelerator.process_index}.jsonl"
        )

        # 在开始前，确保临时文件是空的
        # 'w' 模式会覆盖已存在的文件
        with open(temp_result_file_path, "w", encoding="utf-8") as temp_f:
            pass  # 只是为了清空文件

        with self.accelerator.split_between_processes(data) as inputs:
            logger.info(f"进程 {self.accelerator.process_index} 正在处理 {len(inputs)} 个样本...")
            # results = [] # 不再需要在内存中保存所有结果

            # === 修改：以追加模式('a')打开临时文件 ===
            with open(temp_result_file_path, "a", encoding="utf-8") as temp_f:
                for item in tqdm(inputs, desc=f"Process {self.accelerator.process_index}"):
                    # 直接构造输出文件的路径，不再生成
                    item["output_wav_path"] = os.path.join(
                        generated_audio_dir, f"{item['uid']}.wav"
                    )

                    if not os.path.exists(item["output_wav_path"]):
                        item["output_wav_path"] = os.path.join(
                            generated_audio_dir, f"dialect_{item['uid']}.wav"
                        )

                    if not os.path.exists(item["output_wav_path"]):
                        logger.warning(
                            f"跳过样本 {item['uid']}，因为找不到对应的输出文件: {item['output_wav_path']}"
                        )
                        continue

                    if "api_judge" in metrics:
                        api_response = self._call_dialect_api(item["output_wav_path"])
                        ground_truth_lang = item.get("lang")
                        expected_keyword = LANG_TO_DIALECT_KEYWORD.get(ground_truth_lang)
                        is_correct = (
                            1 if expected_keyword and expected_keyword in api_response else 0
                        )
                        if is_correct == 1:
                            logger.info("API判断为正确")
                        item["api_response"] = api_response
                        item["api_judge_correct"] = is_correct

                    item["transcription"] = item.get("text", "")

                    # === 修改：不再追加到列表，而是直接写入文件 ===
                    temp_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 同步所有进程，确保所有临时文件都已写入完毕
        self.accelerator.wait_for_everyone()

        # --- 修改：主进程负责合并所有临时文件并计算最终结果 ---
        if self.accelerator.is_main_process:
            logger.info("所有进程完成，主进程正在合并临时文件并汇总结果...")

            # === 新增：从所有临时文件中读取结果 ===
            results_gathered = []
            for i in range(self.accelerator.num_processes):
                proc_temp_file = os.path.join(eval_output_dir, f"temp_results_{i}.jsonl")
                try:
                    with open(proc_temp_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                results_gathered.append(json.loads(line.strip()))
                    # 读取后可以删除临时文件，保持目录整洁
                    os.remove(proc_temp_file)
                except FileNotFoundError:
                    logger.warning(f"找不到进程 {i} 的临时文件: {proc_temp_file}")

            metrics_mean = {x: 0.0 for x in metrics if x not in ["wer", "api_judge"]}

            # 计算 'api_judge' 的准确率
            if "api_judge" in metrics:
                total_api_samples = sum(
                    1 for item in results_gathered if "api_judge_correct" in item
                )
                total_api_correct = sum(
                    item.get("api_judge_correct", 0) for item in results_gathered
                )
                api_acc = (
                    (total_api_correct / total_api_samples) * 100 if total_api_samples > 0 else 0
                )
                metrics_mean["api_acc"] = f"{api_acc:.2f}%"

            num_items = len(results_gathered)
            if num_items > 0:
                for k in metrics_mean:
                    if k != "api_acc":
                        metrics_mean[k] /= num_items

            logger.info(f"--- 测试集的平均指标: {metrics_mean} ---")

            # 将最终合并的结果和均值写入文件
            with open(f"{eval_output_dir}/result_summary.jsonl", "w", encoding="utf-8") as f:
                for item in results_gathered:
                    f.write(f"{json.dumps(item, ensure_ascii=False)}\n")
                # 在文件末尾追加均值信息
                f.write(f"{json.dumps({'mean_metrics': metrics_mean}, ensure_ascii=False)}\n")

            end_time = time.time()
            logger.info(
                f"测试集评测完成！耗时: {end_time - start_time:.2f} 秒。结果保存在: {eval_output_dir}"
            )

    def __call__(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/outputs",
        help="保存评测结果的根目录",
    )
    parser.add_argument(
        "--generated_audio_dir",
        type=str,
        default="/root/speech-edit-dataset/eval_scripts/acoustic/api_wavs",
        help="存放已生成音频文件的目录路径",
    )
    args = parser.parse_args()
    # 实例化并运行评测
    evaluation = Dialect_Evaluation(output_dir=args.output_dir)

    evaluation.evaluate_from_folder(generated_audio_dir=args.generated_audio_dir)
