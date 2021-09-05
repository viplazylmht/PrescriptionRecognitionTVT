import torch
import pandas as pd
import os
import uuid
from fuzzywuzzy import fuzz, process

from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename

# detector parameters

UPLOAD_FOLDER = 'static/UPLOAD'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}








app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/")
def index():
    return render_template('policy.html') 

@app.route("/policy")
def policy():
    return render_template('policy.html') 

@app.route("/about_us")
def about_us():
    return render_template('aboutUs.html') 

ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/file-upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        resp = jsonify({"message": "No file part in the request"})
        resp.status_code = 400
        return resp
    file = request.files["file"]
    if file.filename == "":
        resp = jsonify({"message": "No file selected for uploading"})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        print("TRIET FILENAME:" + file.filename)
        filename = secure_filename(file.filename)
        filepath = os.path.join('userImages', str(uuid.uuid4()) + filename)
        file.save(filepath)
        res = readtext(filepath)

        if len(res) > 0:
            print(res)
            resp = jsonify(res)
            resp.status_code = 200
            return resp
        else:
            resp = jsonify({"message": "Not found medical data in your image!"})
            resp.status_code = 400
            return resp
    else:
        resp = jsonify(
            {"message": "Allowed file types are txt, pdf, png, jpg, jpeg, gif"}
        )
        resp.status_code = 400
        return resp
