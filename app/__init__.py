from flask import Flask
from .face_recognition import Model

app = Flask(__name__)

model = Model()

from app import routes
