import base64
import copy
import json
import os

import requests


class GeminiApi:
    url = # 请替换为实际的API URL

    headers = {
        "Content-Type": "application/json",
        # 你需要添加自己的api key
        "Authorization": os.environ.get("Authorization", ""),
        "Cookie": os.environ.get("Cookie", ""),
    }

    payload = {
        "stream": True,
        "model": "gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Please describe the emotion of the audio with firstly "strong" or "normal" and then one of "happy", "neutral", "sad", "disgusted", "angry", "fearful", "surprised", "like", "joyful".',
                    },
                    {"type": "input_audio", "input_audio": {"format": "audio/wav", "data": None}},
                ],
            }
        ],
    }

    @staticmethod
    def audio_to_base64_data_uri(file_path):
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            base64_bytes = base64.b64encode(audio_bytes)
            base64_string = base64_bytes.decode("utf-8")
        data_uri = f"{base64_string}"
        return data_uri

    @staticmethod
    def api_audio_emotion(wav_p):
        data_uri = GeminiApi.audio_to_base64_data_uri(wav_p)
        payload = copy.deepcopy(GeminiApi.payload)
        payload["messages"][0]["content"][1]["input_audio"]["data"] = data_uri
        result = ""
        try:
            response = requests.post(
                GeminiApi.url,
                headers=GeminiApi.headers,
                json=payload,  # 自动序列化 JSON
                stream=False,  # 启用流式传输
            )

            response.raise_for_status()  # 检查 HTTP 错误

            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:{"):
                        json_data = json.loads(decoded_line[5:].strip())
                        result = result + json_data["choices"][0]["delta"]["content"]

            return result

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("result = ", result)
        return result
