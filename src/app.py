import os
import logging
import uuid
import base64
from io import BytesIO

import numpy as np
from flask import Flask, render_template, make_response, request, jsonify, current_app
from PIL import Image

# from waitress import serve

from .runner import runner

logging.basicConfig(level=logging.INFO)

local_testing = False
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
    # print(f"os.getcwd(): {os.getcwd()}")
    temp_filepath = os.path.join(os.getcwd(), api_dir, temp_filename)

    # save image to temp file
    uploaded_file.save(temp_filepath)
    logging.info(f"Saving image file to temp file: {temp_filename}")

    logging.info(f"Running facial recognition...")
    img_array, _, bbox_labels = runner(type='api', input_filepath=api_dir)

    predicted_name = str(bbox_labels[0])

    # Delete temp image file after using it for prediction.
    if not local_testing:
        os.remove(temp_filepath)

    logging.info(f"Temp file deleted: {temp_filename}")

    img = Image.fromarray(img_array)

    #Convert Pillow Image to bytes and then to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue() # bytes
    img_base64 = base64.b64encode(img_byte) #Base64-encoded bytes * not str

    #It's still bytes so json.Convert to str to dumps(Because the json element does not support bytes type)
    img_str = img_base64.decode('utf-8') # str

    resp_dict = {
        "img": img_str,
        "name": predicted_name,
    }
    headers = {"Content-Type": "application/json"}
    return make_response(jsonify(resp_dict), 200, headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below. Then run at the terminal: waitress-serve --port=8000 app:app
    # serve(app, host="0.0.0.0", port=8000)   # using waitress
    # app.run(host="0.0.0.0")
