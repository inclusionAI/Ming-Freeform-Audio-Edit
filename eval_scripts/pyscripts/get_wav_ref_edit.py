# -*- coding: utf-8 -*-

import os
import sys
from tqdm import tqdm

def main():
    """
    主函数，用于处理命令行参数并执行主要逻辑。
    """
    # --- 1. 参数解析 ---
    # 检查命令行参数数量是否正确
    if len(sys.argv) != 4:
        # 打印用法提示并退出
        print(f"用法: python {os.path.basename(__file__)} <lst_file> <wav_dir> <output_file>")
        sys.exit(1)

    wav_dir = sys.argv[2]
    lst_file_path = sys.argv[1]
    output_file_path = sys.argv[3]

    # --- 2. 高效读取并处理 LST 元数据文件 ---
    print(f"正在从 {lst_file_path} 读取元数据...")
    metadata = {}
    try:
        with open(lst_file_path, "r", encoding="utf-8") as f_lst:
            for line in f_lst:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split("|")
                if len(parts) < 5:
                    print(f"警告: LST文件中的行格式不正确，字段少于5个，已跳过: {line}")
                    continue
                
                # 使用第一个字段（uid）作为键，存储整行分割后的列表
                uid = parts[0]
                metadata[uid] = parts
    except FileNotFoundError:
        print(f"错误: 元数据文件 '{lst_file_path}' 不存在。")
        sys.exit(1)
    
    if not metadata:
        print(f"警告: 未从 '{lst_file_path}' 中加载任何元数据。")
        sys.exit(0)
    
    print(f"元数据加载完成，共 {len(metadata)} 条记录。")

    # --- 3. 遍历 WAV 目录并生成输出文件 ---
    try:
        # 获取输入目录中所有以 .wav 结尾的文件名
        wav_filenames = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    except FileNotFoundError:
        print(f"错误: WAV 目录 '{wav_dir}' 不存在。")
        sys.exit(1)

    if not wav_filenames:
        print(f"警告: 在 '{wav_dir}' 目录中未找到任何 .wav 文件。")
        sys.exit(0)

    print(f"在 '{wav_dir}' 中找到 {len(wav_filenames)} 个 .wav 文件，开始处理...")

    # 以写入模式打开输出文件
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        # 使用 tqdm 显示处理进度
        for filename in tqdm(wav_filenames, desc="正在生成输出文件"):
            # 从文件名中获取 uid（不含.wav后缀）
            uid = os.path.splitext(filename)[0]

            # 在元数据字典中查找 uid
            if uid in metadata:
                # 获取主 WAV 文件的绝对路径
                primary_wav_path = os.path.abspath(os.path.join(wav_dir, filename))
                
                # 从元数据中获取所需字段
                meta_parts = metadata[uid]
                prompt_wav = meta_parts[1]  # 第二个字段
                text_content = meta_parts[4] # 第五个字段
                
                # 根据要求，text_label 和 edited_text 都使用第五个字段
                text_label = text_content
                edited_text = text_content
                
                # 组合输出行
                out_line = "|".join([primary_wav_path, prompt_wav, text_label, edited_text])
                
                # 将处理好的一行写入文件
                f_out.write(out_line + "\n")
            else:
                # 如果在元数据中找不到对应的 uid，则打印警告
                print(f"警告: 在 {lst_file_path} 中未找到 UID '{uid}' 的记录，已跳过文件 {filename}。")

    print(f"\n处理完成。输出结果已保存至: {output_file_path}")

if __name__ == "__main__":
    main()
