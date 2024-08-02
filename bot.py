import os
import time
import logging
import csv
import threading
import random
import base64
from typing import Dict, Any, Optional
import telebot
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers.decoders import ByteLevel
from kandinsky import Kandinsky


TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL_ID = "@polls4life"

KANDINSKY_API_KEY = os.environ["KANDINSKY_API_KEY"]
KANDINSKY_SECRET_KEY = os.environ["KANDINSKY_SECRET_KEY"]
KANDINSKY_IMAGE_WIDTH = 1024
KANDINSKY_IMAGE_HEIGHT = 1024
KANDINSKY_IMAGE_STYLE = "DEFAULT"

MODEL_NAME = "./finetuned_model_medium_3"  # "./finetuned_model_small_4_5"
MAX_NEW_TOKENS = 400
DEVICE = "cpu"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
HISTORY_FILE_NAME = "compares.csv"


like_smiles = [
    "😘",
    "😊",
    "❤️️",
    "🥳",
    "🥰",
    "🙏",
    "💋",
    "😉",
    "🤩",
    "😎",
    "🤗",
    "🤭",
    "😁",
    "😄",
    "😇",
    "🤝",
    "😀",
    "😋",
    "🍆",
]


bot = telebot.TeleBot(TELEGRAM_TOKEN)
is_working = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
decoder = ByteLevel()
lock = threading.Lock()
df = pd.read_csv("important_polls_cleared_updated.csv")

user_polls = {}
kandinsky = Kandinsky(KANDINSKY_API_KEY, KANDINSKY_SECRET_KEY)
logger = logging.getLogger(__name__)


def logging_configure() -> None:
    formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
    )
    file_handler = logging.FileHandler("server.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_random_decoding_config() -> Dict[str, Any]:
    config = {}
    config["max_new_tokens"] = MAX_NEW_TOKENS
    if random.random() < 0.5:
        config["no_repeat_ngram_size"] = random.sample(
            population=[3, 4, 5, 6], k=1, counts=[1, 8, 4, 2]
        )[0]
    config["do_sample"] = True
    config["top_k"] = random.sample(
        population=[2, 4, 8, 16, 32, 50, 64, 100],
        k=1,
        counts=[1, 1, 1, 1, 2, 14, 1, 1],
    )[0]
    config["top_p"] = random.sample(
        population=[0, 0.85, 0.9, 0.92, 0.95, 0.96],
        k=1,
        counts=[4, 1, 1, 1, 12, 1],
    )[0]
    config["temperature"] = random.sample(
        population=[0.5, 0.8, 1.0, 1.2], k=1, counts=[1, 2, 8, 1]
    )[0]
    return config


def get_default_decoding_config() -> Dict[str, Any]:
    config = {}
    config["max_new_tokens"] = MAX_NEW_TOKENS
    config["no_repeat_ngram_size"] = 10
    config["do_sample"] = True
    config["top_k"] = 50
    config["top_p"] = 0.95
    return config


def poll_is_small_enough(question: str, options: list[str]) -> bool:
    return len(question) <= 300 and all(
        len(option) <= 100 for option in options
    )


def generate_polls(
    prompt: str = "<s>",
    num_return_sequences=1,
    decoding_config: Optional[Dict[str, Any]] = None,
) -> list[str]:
    if decoding_config is None:
        decoding_config = get_default_decoding_config()
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
    model_outputs = model.generate(
        **model_inputs,
        **decoding_config,
        num_return_sequences=num_return_sequences,
    )
    return tokenizer.batch_decode(model_outputs, skip_special_tokens=True)


def postproccess(question: list[str]) -> str:
    question[-1] = f"**{question[-1]}**"
    return "\n".join(question)


@bot.message_handler(commands=["start"])
def start(_):
    global is_working
    with lock:
        is_working = True
    while is_working:
        poll = generate_polls()[0]
        logger.info("%s", poll)

        *question, option1, option2 = poll.split("\n")
        question = "\n".join(question)

        logger.info("Started to generate image")
        content, censored = kandinsky.generate(
            prompt=poll,
            width=KANDINSKY_IMAGE_WIDTH,
            height=KANDINSKY_IMAGE_HEIGHT,
            style=KANDINSKY_IMAGE_STYLE,
        )
        logger.info("Generated image")

        if not censored:
            bot.send_photo(TELEGRAM_CHANNEL_ID, base64.b64decode(content))
        if poll_is_small_enough(question, [option1, option2]):
            bot.send_poll(
                TELEGRAM_CHANNEL_ID,
                question,
                [option1.lstrip("1."), option2.lstrip("2.")],
            )
        else:
            bot.send_message(
                TELEGRAM_CHANNEL_ID,
                poll,
                parse_mode="markdown",
            )
            bot.send_poll(
                TELEGRAM_CHANNEL_ID,
                "👆",
                ["1", "2"],
            )
        time.sleep(5 * 60)


@bot.message_handler(commands=["stop"])
def stop(_):
    global is_working
    with lock:
        is_working = False


def construct_poll_from_sample(sample: pd.DataFrame) -> str:
    question = sample["question"].item().strip()
    if not question.strip(".").endswith("?"):
        question = question + "\nЧто бы Вы выбрали?"
    answer1 = sample["answer1"].item()
    answer2 = sample["answer2"].item()
    return f"{BOS_TOKEN}{question}\n1. {answer1}\n2. {answer2}{EOS_TOKEN}"


@bot.message_handler(commands=["dev"])
def compare_generation(message):
    existing_poll = construct_poll_from_sample(df.sample(n=1))
    tokens = tokenizer.tokenize(existing_poll)
    tokens = tokens[: 1 + random.randint(0, len(tokens) - 1) // 8]
    prompt = decoder.decode(tokens)
    user_polls[message.chat.id] = generate_polls(
        prompt=prompt, num_return_sequences=2
    )

    markup = telebot.types.InlineKeyboardMarkup()
    markup.row(
        telebot.types.InlineKeyboardButton("Опрос 1", callback_data="option1"),
        telebot.types.InlineKeyboardButton("Опрос 2", callback_data="option2"),
        telebot.types.InlineKeyboardButton("Эскобар", callback_data="option3"),
    )
    bot.send_message(message.chat.id, user_polls[message.chat.id][0])
    bot.send_message(
        message.chat.id,
        user_polls[message.chat.id][1],
        reply_markup=markup,
    )


@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    polls = user_polls[call.message.chat.id]
    if call.data == "option1":
        print(polls[0])
        save_history(*polls, first_is_better=True)
    elif call.data == "option2":
        print(polls[1])
        save_history(*polls, first_is_better=False)
    elif call.data == "option3":
        print("Эскобар выбран!")
    bot.answer_callback_query(call.id, "Выбран вариант " + call.data)
    bot.send_message(
        call.message.chat.id,
        like_smiles[random.randint(0, len(like_smiles) - 1)],
    )
    bot.send_message(
        call.message.chat.id,
        "Генерю новые опросы...",
    )
    compare_generation(call.message)


def save_history(prompt1, prompt2, first_is_better):
    with lock:
        with open(
            HISTORY_FILE_NAME, "a", newline="", encoding="utf8"
        ) as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=",",
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow([prompt1, prompt2, first_is_better])


if __name__ == "__main__":
    logging_configure()
    logger.info("Model loaded")
    bot.polling(none_stop=True, timeout=10000)
