import argparse
import os
import re


def process_log_file(input_path, output1_path, output2_path):

    line_regex = re.compile(
        r"^(\S+\.wav)\s+" r"[\d.-]+\s+" r"(.+?)\s+" r"((?:[\d.-]+\s*){2}[\d.-]+)$"
    )

    processed_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output1_path, "w", encoding="utf-8"
    ) as outfile1, open(output2_path, "w", encoding="utf-8") as outfile2:
        for line in infile:
            line = line.strip()

            if not line or line.startswith("utt") or line.startswith("WER:"):
                continue

            match = line_regex.match(line)
            if not match:
                fallback_regex = re.compile(r"^(\S+\.wav)\s+[\d.-]+\s+(.*)")
                match_fb = fallback_regex.match(line)
                if match_fb:
                    full_path = match_fb.group(1)
                    full_text_block = match_fb.group(2).strip()
                else:
                    print(f"警告: 无法解析该行 -> {line}")
                    continue
            else:
                full_path = match.group(1)
                full_text_block = match.group(2).strip()

            text_parts = full_text_block.split(None, 1)

            text_res = text_parts[0]
            if len(text_parts) > 1:
                text_ref_with_spaces = text_parts[1].strip()
            else:
                text_ref_with_spaces = ""

            filename = os.path.basename(full_path).replace(".wav", "")

            outfile1.write(f"{filename} {text_res}\n")

            if text_ref_with_spaces:
                text_ref_no_spaces = text_ref_with_spaces.replace(" ", "")
                outfile2.write(f"{filename} {text_ref_no_spaces}\n")
            else:
                outfile2.write(f"{filename} \n")

            processed_count += 1


def process_log_file_new(input_path, output1_path, output2_path):

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output1_path, "w", encoding="utf-8"
    ) as outfile1, open(output2_path, "w", encoding="utf-8") as outfile2:

        for line in infile:
            line = line.strip()

            if not line or line.startswith("utt") or line.startswith("WER:"):
                continue

            row = line.split("\t")

            filename = row[0].split("/")[-1].replace(".wav", "")
            outfile1.write(f"{filename} {row[2]}\n")
            outfile2.write(f"{filename} {row[3]}\n")


def main():
    parser = argparse.ArgumentParser(description="解析日志并生成两个格式化的输出文件。")
    parser.add_argument("input_file", help="输入的推理结果文件路径。")
    parser.add_argument("output_file1", help="第一个输出文件的路径。")
    parser.add_argument("output_file2", help="第二个输出文件的路径。")

    args = parser.parse_args()
    process_log_file_new(args.input_file, args.output_file1, args.output_file2)


if __name__ == "__main__":
    main()
