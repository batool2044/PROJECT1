import cv2
import face_recognition
import pickle


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
        results = face_recognition.compare_faces(
            [encoding], unknown_encoding, tolerance)
        if results[0]:
            return name  # Return the name of the matched face
    return None


haarcascade = "C:\\Users\\ONE\\wook\\project1\\haarcascade_frontalface_default.xml"
database_file = "C:\\Users\\ONE\\wook\\project1\\database"

database = load(database_file)

video_capture = cv2.VideoCapture(
    "C:\\Users\\ONE\\Downloads\\New folder\\download (1).jpg")
face_cascade = cv2.CascadeClassifier(haarcascade)
height = 300
width = 400
while True:
    tr, image = video_capture.read()
    if not tr:
        print("Error: Failed to read frame.")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (top, right, bottom, left) in faces:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        name = input("enter the name\n")
        cv2.putText(image, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rgb_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)
        if face_encodings:
            for encoding in face_encodings:
                database[name] = encoding
            save(database, database_file)

    resized_image = cv2.resize(image, (width, height))
    cv2.imshow('Detected Faces', resized_image)

    if cv2.waitKey(10000) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
