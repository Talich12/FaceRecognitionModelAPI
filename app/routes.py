from app import app, model

from flask import jsonify
from flask import request

from PIL import Image
from io import BytesIO

@app.route("/predict", methods=["POST"])
def predict():

    img = request.files['file'].read()  # read the file
    img = Image.open(BytesIO(img))  # convert byte stream to PIL Image

    name = model.predict_face_by_img(img)
    return jsonify(name=name)

@app.route("/first", methods=["GET"])
def first():
    return jsonify(count="Pasha PIDOR")
