import os
import logging
import uuid
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template, make_response, request, jsonify, current_app
from PIL import Image

from .runner import runner

logging.basicConfig(level=logging.INFO)

api_dir = "src/data/temp_api"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return current_app.send_static_file("indexpage.html")


@app.route("/fr", methods=["POST"])
def fr():
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return render_template("predict.html")

    logging.info(f"Received input image file: {uploaded_file.filename}")
    # create temporary file with random name
    temp_filename_stem = str(uuid.uuid4())
    _, file_extension = os.path.splitext(uploaded_file.filename)
    temp_filename = temp_filename_stem + file_extension

    # create temp dir for this specific API call, so that we don't have 
    # multiple images in the main temp dir and suffer data leakage.
    temp_api_dir = os.path.join(api_dir, temp_filename_stem)
    logging.info(f"Temp dir for this API call: {temp_api_dir}")
    Path(temp_api_dir).mkdir(parents=True, exist_ok=True)

    temp_filepath = os.path.join(temp_api_dir, temp_filename)

    # save image to temp file
    uploaded_file.save(temp_filepath)
    logging.info(f"Saving image file to temp file: {temp_filename}")

    logging.info(f"Running facial recognition...")
    img_array, _, bbox_labels = runner(type="api", input_filepath=temp_api_dir)

    predicted_name = str(bbox_labels[0])

    # delete temp image file and its associated temp dir
    os.remove(temp_filepath)
    os.rmdir(temp_api_dir)

    logging.info(f"Temp file deleted: {temp_filename}")

    img = Image.fromarray(img_array)

    # convert Pillow Image binary data to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()  # bytes
    img_base64 = base64.b64encode(
        img_byte
    )  # base64-encoded bytes (type is still byte, not str)

    # convert to str because json does not support bytes type
    img_str = img_base64.decode("utf-8")  # str

    resp_dict = {
        "img": img_str,
        "name": predicted_name,
    }
    headers = {"Content-Type": "application/json"}
    return make_response(jsonify(resp_dict), 200, headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below.
    # app.run(host="0.0.0.0")
