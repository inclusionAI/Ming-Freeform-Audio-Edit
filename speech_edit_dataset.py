# -*- coding: utf-8 -*-
import csv
import os

import datasets

_CITATION = """\
@inproceedings{your_name_2023_speech_edit,
  title={A Dataset for Speech Editing across Multiple Languages and Tasks},
  author={Your Name and Co-authors},
  booktitle={Proceedings of a Conference},
  year={2025}
}
"""

_DESCRIPTION = """\
这是一个用于语音编辑任务的测试集，涵盖中、英文两种语言，
涉及三种编辑任务：增加（insertion）、删除（deletion）和替换（substitution）。
每个样本都包含原始音频、音频路径、描述编辑操作的自然语言指令、原始文本转录，以及编辑后的目标文本转录。
"""

_HOMEPAGE = "https://huggingface.co/datasets/your-username/your-dataset-name"
_LICENSE = "CC-BY-SA-4.0"


class SpeechEditConfig(datasets.BuilderConfig):
    """
    SpeechEditConfig 继承自 BuilderConfig，用于定义每个子数据集的特定配置。
    - language: 语言 (e.g., 'en', 'zh')
    - task: 任务类型 (e.g., 'insertion', 'deletion')
    - data_file: 与此配置关联的元数据文件名
    """

    def __init__(self, language=None, task=None, data_file=None, **kwargs):
        super().__init__(**kwargs)
        self.language = language
        self.task = task
        self.data_file = data_file


class SpeechEditDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SpeechEditConfig(
            name="en_insertion",
            language="en",
            task="insertion",
            data_file="meta/ins/meta_en_insertion.csv",
            description="英文增加任务 (English Insertion Task)",
        ),
        SpeechEditConfig(
            name="en_deletion",
            language="en",
            task="deletion",
            data_file="meta/del/meta_en_deletion.csv",
            description="英文删除任务 (English Deletion Task)",
        ),
        SpeechEditConfig(
            name="en_substitution",
            language="en",
            task="substitution",
            data_file="meta/sub/meta_en_substitution.csv",
            description="英文替换任务 (English Substitution Task)",
        ),
        SpeechEditConfig(
            name="zh_insertion",
            language="zh",
            task="insertion",
            data_file="meta/ins/meta_zh_insertion.csv",
            description="中文增加任务 (Chinese Insertion Task)",
        ),
        SpeechEditConfig(
            name="zh_deletion",
            language="zh",
            task="deletion",
            data_file="meta/del/meta_zh_deletion.csv",
            description="中文删除任务 (Chinese Deletion Task)",
        ),
        SpeechEditConfig(
            name="zh_substitution",
            language="zh",
            task="substitution",
            data_file="meta/sub/meta_zh_substitution.csv",
            description="中文替换任务 (Chinese Substitution Task)",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "instruction": datasets.Value("string"),
                    "original_text": datasets.Value("string"),
                    "edited_text": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "task": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    # 指定从哪里读取数据文件
    def _split_generators(self, dl_manager):
        data_path = dl_manager.download_and_extract(self.config.data_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath):
        """从元数据文件中读取数据并生成样本"""

        audio_dir = "wavs"

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")

            for idx, row in enumerate(reader):
                uid = row["file_name"]
                abs_audio_path = row["path"]
                instruction = row["instruction"]
                original_text = row["original_text"]
                edited_text = row["edited_text"]

                audio_filename = os.path.basename(abs_audio_path)
                relative_audio_path = os.path.join(audio_dir, audio_filename)

                yield idx, {
                    "uid": uid,
                    "audio": relative_audio_path,
                    "instruction": instruction,
                    "original_text": original_text,
                    "edited_text": edited_text,
                    "language": self.config.language,
                    "task": self.config.task,
                }
