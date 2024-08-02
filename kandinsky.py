import json
import time
import threading
import csv
import os.path
import base64

import requests

HISTORY_FILE_NAME = "image_history.csv"


class Text2ImageAPI:
    def __init__(self, url, api_key, secret_key):
        self.url = url
        self.auth_headers = {
            "X-Key": f"Key {api_key}",
            "X-Secret": f"Secret {secret_key}",
        }

    def get_model(self):
        try:
            response = requests.get(
                self.url + "key/api/v1/models",
                headers=self.auth_headers,
                timeout=240,
            )
            data = response.json()
        except Exception:
            return 0
        return data[0]["id"]

    # https://cdn.fusionbrain.ai/static/styles/api
    def generate(self, prompt, model, width, height, style, images=1):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "style": style,
            "negativePromptUnclip": "Текст, надписи",
            "generateParams": {"query": f"{prompt}"},
        }

        data = {
            "model_id": (None, model),
            "params": (None, json.dumps(params), "application/json"),
        }
        try:
            response = requests.post(
                self.url + "key/api/v1/text2image/run",
                headers=self.auth_headers,
                files=data,
                timeout=240,
            )
            data = response.json()
        except Exception:
            return ""
        if "uuid" in data:
            return data["uuid"]
        return ""  # ?

    def check_generation(self, request_id, attempts=40, delay=5):
        while attempts > 0:
            try:
                response = requests.get(
                    self.url + "key/api/v1/text2image/status/" + request_id,
                    headers=self.auth_headers,
                    timeout=240,
                )
                data = response.json()
            except Exception:
                return None, None
            if data["status"] == "DONE":
                return data["images"], data["censored"]
            print(data["status"])

            attempts -= 1
            time.sleep(delay)
        return None, None


class Kandinsky:

    def __init__(
        self, api_key: str, secret_key: str, keep_history: bool = False
    ):
        self.keep_history = keep_history
        self.api = Text2ImageAPI(
            "https://api-key.fusionbrain.ai/",
            api_key,
            secret_key,
        )
        self.model_id = self.api.get_model()
        self.lock = threading.Lock()
        if not os.path.exists(HISTORY_FILE_NAME):
            with open(
                HISTORY_FILE_NAME, "a", newline="", encoding="utf8"
            ) as file:
                writer = csv.writer(
                    file,
                    delimiter=",",
                    quotechar="|",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writerow(
                    ["prompt", "width", "height", "style", "image_name"]
                )

    def save_history(self, prompt, width, height, style, image_data):
        with self.lock:
            with open(HISTORY_FILE_NAME, "r", encoding="utf8") as file:
                for count, _ in enumerate(file):
                    pass

            image_name = "image" + str(count) + ".jpg"
            with open(
                HISTORY_FILE_NAME, "a", newline="", encoding="utf8"
            ) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=",",
                    quotechar="|",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writerow([prompt, width, height, style, image_name])
        with open(os.path.join("images", image_name), "wb") as file:
            file.write(image_data)

    def generate(self, prompt: str, width=1024, height=1024, style="DEFAULT"):
        uuid = self.api.generate(
            prompt,
            self.model_id,
            width=width,
            height=height,
            style=style,
            images=1,
        )
        images, censored = self.api.check_generation(uuid)
        if images is not None:
            if self.keep_history:
                image_data = base64.b64decode(images[0])
                save_thread = threading.Thread(
                    target=self.save_history,
                    args=(prompt, width, height, style, image_data),
                )
                save_thread.start()
            return images[0], censored
        return None, None
