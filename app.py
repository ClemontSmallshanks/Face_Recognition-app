import os
import cv2
import faiss
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from mtcnn import MTCNN
from deepface import DeepFace
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

FAISS_INDEX_PATH = "faiss_index.faiss"
NAMES_PATH = "names.pkl"
DIM = 512  # ArcFace embedding size

# ----------- Utility Functions ----------- #
def align_face(image, face):
    x, y, w, h = face["box"]
    return image[y:y+h, x:x+w]

def extract_embedding_from_image_array(image_array):
    # DeepFace expects RGB images
    if image_array is None or image_array.size == 0:
        return None
    rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    try:
        embedding = DeepFace.represent(
            img_path=rgb_img, model_name="ArcFace", enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding, dtype="float32")
    except Exception:
        return None

def load_faiss_db():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(NAMES_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(NAMES_PATH, "rb") as f:
            names = pickle.load(f)
        return index, names
    else:
        return faiss.IndexFlatL2(DIM), []

def save_faiss_db(index, names):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)

def reset_db():
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(NAMES_PATH):
        os.remove(NAMES_PATH)

# ----------- Flask Routes ----------- #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/add", methods=["GET", "POST"])
def add_person():
    if request.method == "POST":
        name = request.form.get("name")
        files = request.files.getlist("images")
        if not name or not files:
            return render_template("add_person.html", error="Name and images required.")

        detector = MTCNN()
        index, names = load_faiss_db()

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
                file.save(img_path)

                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)
                if faces:
                    aligned = align_face(img, faces[0])
                    emb = extract_embedding_from_image_array(aligned)
                    if emb is not None:
                        index.add(emb.reshape(1, -1))
                        names.append(name)

        save_faiss_db(index, names)
        return redirect(url_for("home"))

    return render_template("add_person.html")

@app.route("/recognize", methods=["GET", "POST"])
def recognize_faces():
    recognized_names = []
    output_image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
            file.save(img_path)

            img = cv2.imread(img_path)
            if img is None:
                return render_template("recognize.html", names=[], image_url=None, error="Invalid image.")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detector = MTCNN()
            faces = detector.detect_faces(img_rgb)

            index, names = load_faiss_db()

            for face in faces:
                aligned = align_face(img, face)
                emb = extract_embedding_from_image_array(aligned)
                if emb is not None and index.ntotal > 0:
                    D, I = index.search(emb.reshape(1, -1), 1)
                    matched_name = names[I[0][0]] if D[0][0] < 1.0 else "Unknown"
                else:
                    matched_name = "Unknown"

                recognized_names.append(matched_name)

                # Draw bounding box + name
                x, y, w, h = face["box"]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, matched_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Save annotated image
            out_filename = f"result_{timestamp}.jpg"
            out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
            cv2.imwrite(out_path, img)
            output_image_url = url_for("static", filename=f"uploads/{out_filename}")

    return render_template("recognize.html", names=recognized_names, image_url=output_image_url)

@app.route("/reset_db", methods=["POST"])
def reset_database():
    reset_db()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)