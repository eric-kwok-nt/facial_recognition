import logging
import argparse
import os
import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw

logging.basicConfig(level=logging.INFO)

endpoint = "http://localhost:8000/fr"
# endpoint = "http://0.0.0.0:8000/fr"
# endpoint = 'http://team4.aiap.okdapp.tekong.aisingapore.net/fr'


def get_prediction(image_path, endpoint=endpoint):
    files = {"file": open(os.path.join(os.getcwd(), image_path), "rb")}
    r = requests.post(endpoint, files=files)

    # raise exception if HTTP response status is not 200
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    prediction = r.json()

    img = prediction["img"] # get image data (base64 in utf-8-encoded str)
    img = base64.b64decode(img) # convert base64 to original binary data bytes
    img = BytesIO(img)
    img = Image.open(img)

    # Convert from BGR to RGB
    b, g, r = img.split()
    img = Image.merge('RGB', (r, g, b))

    img.show()
    print(f"Label: {prediction['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    get_prediction(args.image)
