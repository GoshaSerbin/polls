import threading
import csv
import os.path
from typing import Tuple
from openai import OpenAI

HISTORY_FILE_NAME = "gpt_history.csv"


class GPT:
    def __init__(self, tokens_limit=800, model_name=None):
        if model_name is None:  # gpt-3.5-turbo-1106 # gpt-4-turbo
            model_name = "gpt-3.5-turbo-1106"
        self.model_name = model_name
        self._client = OpenAI(
            api_key="",
            base_url="",
        )
        self.tokens_limit = tokens_limit
        self.lock = threading.Lock()
        if not os.path.exists(HISTORY_FILE_NAME):
            with open(HISTORY_FILE_NAME, "a", newline="") as file:
                writer = csv.writer(
                    file,
                    delimiter=",",
                    quotechar="|",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writerow(["messages", "content", "finish_reason"])

    def save_history(self, messages, chat_completion):
        with self.lock, open(HISTORY_FILE_NAME, "a", newline="") as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=",",
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(
                [
                    messages,
                    chat_completion.choices[0].message.content,
                    chat_completion.choices[0].finish_reason,
                ]
            )

    def create(
        self,
        messages: list[dict],
        max_tokens,
        temperature=1.2,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    ) -> Tuple[str, str]:

        max_tokens = min(max_tokens, self.tokens_limit)
        chat_completion = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        # Запускаем сохранение сообщения в отдельном потоке
        save_thread = threading.Thread(
            target=self.save_history,
            args=(
                messages,
                chat_completion,
            ),
        )
        save_thread.start()
        return (
            chat_completion.choices[0].message.content,
            chat_completion.choices[0].finish_reason,
        )
