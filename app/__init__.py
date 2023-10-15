from flask import Flask
from .face_recognition import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app=app)
model = Model()

from app import routes
