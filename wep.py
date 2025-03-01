from flask import Flask, request, render_template
import cv2
import face_recognition
import pickle
import os
import numpy as np

app = Flask(__name__)
database_file = "C:\\Users\\ONE\\wook\\project1\\database"

def save(database, database_file):
    with open(database_file, 'wb') as f:
        pickle.dump(database, f)

def load(database_file):
    try:
        with open(database_file, 'rb') as f:
            encoding_data = pickle.load(f)
            return encoding_data
    except (EOFError, FileNotFoundError):
        return {}

def compare_face(known_encodings, unknown_encoding, tolerance=0.6):
    for name, encoding in known_encodings.items():
        results = face_recognition.compare_faces([encoding], unknown_encoding, tolerance)
        if results[0]:
            return name
    return None

@app.route('/compare', methods=['POST'])
def compare():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    img = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(img)

    if not face_encodings:
        return 'No face detected in the uploaded image.'


    database = load(database_file)

    name = compare_face(database, face_encodings[0])

    if name:
            return f'WELCOME {name}'
    else:
            return 'Sorry ,plase CREATE ACCOUNT'

@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    img = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(img)

    if not face_encodings:
        return 'No face detected in the uploaded image.'

    database = load(database_file)

    username = request.form['username']

    database[username] = face_encodings[0]

    save(database, database_file)

    return f"User {username} registered successfully!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

if __name__ == '__main__':
    app.run(debug=True)
