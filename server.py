from flask import Flask, request, jsonify
import json
import time
from openai import OpenAI
import csv
import logging

from gpt import GPT
from kandinsky import Kandinsky

HOST_NAME = "146.185.211.97"
PORT = "8093"
DEBUG_MODE = False

app = Flask(__name__)
gpt = GPT() # model_name ="gpt-4o"
kandinsky = Kandinsky()
iter = 1
logger = logging.getLogger(__name__)

def logging_configure():
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

@app.route("/talk", methods=["POST"])
def handle_talk_request():
    global iter
    iter += 1
    message = request.form["message"]
    max_tokens = int(request.form["max_tokens"])
    temperature = float(request.form["temperature"].replace(',','.'))
    logger.info("Server received (message: %s, max_tokens: %i, temperature: %f)", message, max_tokens, temperature)
    prompt = json.loads(message)
    if DEBUG_MODE:
        return "Да, Debug mode is on"
    content, finish_reason = gpt.create(messages=prompt, max_tokens=max_tokens, temperature=temperature)
    logger.info("Answer is (content: %s, finish_reason: %s)", content, finish_reason)
    return content

@app.route("/image", methods=["POST"])
def handle_image_request():


    global iter
    iter += 1
    prompt = request.form["prompt"]
    style = request.form["style"]
    width = int(request.form["width"])
    height = int(request.form["height"])
    logger.info("Server received (prompt: %s, style: %s, w: %i, h: %i)", prompt, style, width, height)

    if DEBUG_MODE:
        with open('debug_image.jpg', 'rb') as file:
            return file.read()
    
    content, censored = kandinsky.generate(prompt=prompt, width=width, height=height, style=style)
    response = {
        "data": content,
        "censored": censored
    }
    return jsonify(response)


if __name__ == "__main__":
    from waitress import serve

    logging_configure()
    serve(app, host=HOST_NAME, port=PORT)