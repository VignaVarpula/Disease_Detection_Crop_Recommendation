

# from flask import Flask, render_template, request, send_from_directory
# import tensorflow as tf
# import numpy as np
# import cv2, os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # -------------------------------
# # Load Disease Detection Model
# # -------------------------------
# MODEL_PATH = r"C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras"
# model = tf.keras.models.load_model(MODEL_PATH)

# class_names = [
#     "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
#     "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
#     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
#     "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
#     "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
#     "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
#     "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
#     "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
#     "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
#     "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
# ]

# def preprocess_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (128, 128))
#     img = np.expand_dims(img, axis=0)
#     return img

# # -------------------------------
# # Routes
# # -------------------------------
# @app.route("/")
# def home():
#     return render_template("index.html")  # Main selection page

# @app.route("/disease", methods=["GET", "POST"])
# def disease_detection():
#     if request.method == "POST":
#         if "image" not in request.files:
#             return render_template("disease.html", error="No file uploaded")

#         file = request.files["image"]
#         if file.filename == "":
#             return render_template("disease.html", error="No file selected")

#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)

#         img = preprocess_image(file_path)
#         preds = model.predict(img, verbose=0)
#         pred_index = np.argmax(preds)
#         confidence = float(np.max(preds) * 100)
#         prediction = class_names[pred_index]

#         if confidence < 50:
#             prediction = "Uncertain / Unknown"

#         return render_template(
#             "disease.html",
#             filename=filename,
#             prediction=prediction,
#             confidence=confidence
#         )

#     return render_template("disease.html")

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# # (Placeholder) — crop recommendation page
# @app.route("/crop")
# def crop_page():
#     return render_template("crop.html")

# # -------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
import cv2, os
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load Disease Detection Model
# -------------------------------
MODEL_PATH = r"C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# -------------------------------
# Load Crop Recommendation Model
# -------------------------------
# Load and train lightweight model from crop_train.py logic
# Load and train lightweight model from crop_train.py logic
data = pd.read_csv(r"C:\Youtube\DiseaseDetection\dataset\Crop_recommendation.csv")
merged = data.drop_duplicates(subset=['N','P','K','temperature','humidity','ph','rainfall'])


X = merged[['N','P','K','temperature','humidity','ph','rainfall']]
y = merged['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_crop = SVC(kernel='linear', random_state=42)
model_crop.fit(X_scaled, y)

def predict_crop(features):
    features_df = pd.DataFrame([features])
    scaled_features = scaler.transform(features_df)
    prediction = model_crop.predict(scaled_features)
    return prediction[0]

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/disease", methods=["GET", "POST"])
def disease_detection():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("disease.html", error="No file uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("disease.html", error="No file selected")

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        preds = model.predict(img, verbose=0)
        pred_index = np.argmax(preds)
        confidence = float(np.max(preds) * 100)
        prediction = class_names[pred_index]

        if confidence < 50:
            prediction = "Uncertain / Unknown"

        return render_template(
            "disease.html",
            filename=filename,
            prediction=prediction,
            confidence=confidence
        )

    return render_template("disease.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------------------
# Crop Recommendation Page
# -------------------------------
@app.route("/crop", methods=["GET", "POST"])
def crop_page():
    if request.method == "POST":
        try:
            features = {
                "N": float(request.form["N"]),
                "P": float(request.form["P"]),
                "K": float(request.form["K"]),
                "temperature": float(request.form["temperature"]),
                "humidity": float(request.form["humidity"]),
                "ph": float(request.form["ph"]),
                "rainfall": float(request.form["rainfall"])
            }

            crop = predict_crop(features)
            return render_template("crop.html", crop=crop, features=features)

        except Exception as e:
            return render_template("crop.html", error=f"Error: {str(e)}")

    return render_template("crop.html")

# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
