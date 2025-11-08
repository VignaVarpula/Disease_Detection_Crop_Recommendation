
# #------------------------------------ test_predictions_fixed.csv predict correctly for test but not real iamges---
# # diagnostic_predict.py
# import os
# import json
# import numpy as np
# import csv
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.utils import register_keras_serializable
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ---------------------------
# # Register preprocess_input so Keras can load Lambda layer
# # ---------------------------
# @register_keras_serializable(package="EfficientNet")
# def preprocess_input_registered(x):
#     return preprocess_input(x)

# # ---------------------------
# # Paths
# # ---------------------------
# BASE_DIR = r"C:\Youtube\DiseaseDetection"
# MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model_transfer.keras")
# CLASS_JSON = os.path.join(BASE_DIR, "class_names_transfer.json")
# TEST_DIR = os.path.join(BASE_DIR, "dataset", "real_world_test")
# OUT_CSV = os.path.join(BASE_DIR, "real_test_predictions_fixed.csv")

# # ---------------------------
# # Load model and class names
# # ---------------------------
# print("Loading model with registered Lambda(preprocess_input)...")
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input_registered})
# print("✅ Model loaded successfully.")

# with open(CLASS_JSON, 'r', encoding='utf-8') as f:
#     class_names = json.load(f)
# print(f"Loaded {len(class_names)} class labels.")

# # ---------------------------
# # Prediction function
# # ---------------------------
# def predict_image(img_path, image_size=(180,180)):
#     img = Image.open(img_path).convert("RGB").resize(image_size)
#     arr = np.expand_dims(np.array(img), axis=0)  # raw 0–255 input
#     preds = model.predict(arr, verbose=0)[0]
#     idx = np.argmax(preds)
#     conf = preds[idx] * 100
#     return class_names[idx], conf, preds

# # ---------------------------
# # Run predictions
# # ---------------------------
# rows = []
# img_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]

# if not img_files:
#     raise SystemExit("❌ No images found in test directory!")

# print(f"Predicting on {len(img_files)} test images...")

# for i, file in enumerate(sorted(img_files)):
#     path = os.path.join(TEST_DIR, file)
#     label, conf, preds = predict_image(path)
#     rows.append([file, label, f"{conf:.2f}%"])
#     if i < 5:  # show some sample top predictions
#         print(f"{file}: {label} ({conf:.2f}%)")

# # Save results to CSV
# with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Filename", "PredictedLabel", "Confidence(%)"])
#     writer.writerows(rows)

# print("✅ Predictions saved to:", OUT_CSV)

# # ---------------------------
# # Optional: Confusion Matrix & Report (if test set has subfolders as class names)
# # ---------------------------
# if os.path.isdir(TEST_DIR) and any(os.path.isdir(os.path.join(TEST_DIR, d)) for d in os.listdir(TEST_DIR)):
#     print("\nEvaluating confusion matrix and classification report...")
#     true_labels, pred_labels = [], []
#     for class_dir in sorted(os.listdir(TEST_DIR)):
#         class_path = os.path.join(TEST_DIR, class_dir)
#         if not os.path.isdir(class_path): continue
#         for file in os.listdir(class_path):
#             if not file.lower().endswith(('.jpg','.jpeg','.png')): continue
#             fpath = os.path.join(class_path, file)
#             pred, conf, _ = predict_image(fpath)
#             true_labels.append(class_dir)
#             pred_labels.append(pred)

#     cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
#     plt.figure(figsize=(12,10))
#     sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
#     plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(os.path.join(BASE_DIR, "confusion_matrix_fixed.png"))
#     print("✅ Confusion matrix saved.")

#     print("\nClassification Report:")
#     print(classification_report(true_labels, pred_labels))
# else:
#     print("\n(No labeled subfolders found — skipped confusion matrix.)")


# diagnostic_predict.py — improved real-world inference

# import os
# import json
# import numpy as np
# import csv
# from PIL import Image, ImageOps
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.utils import register_keras_serializable
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ---------------------------
# # Register preprocess_input
# # ---------------------------
# @register_keras_serializable(package="EfficientNet")
# def preprocess_input_registered(x):
#     return preprocess_input(x)

# # ---------------------------
# # Paths
# # ---------------------------
# BASE_DIR = r"C:\Youtube\DiseaseDetection"
# MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model_transfer.keras")
# CLASS_JSON = os.path.join(BASE_DIR, "class_names_transfer.json")
# TEST_DIR = os.path.join(BASE_DIR, "dataset", "real_world_test")
# OUT_CSV = os.path.join(BASE_DIR, "real_test_predictions_fixed.csv")

# IMAGE_SIZE = (180, 180)

# # ---------------------------
# # Load model & class names
# # ---------------------------
# print("Loading model with registered Lambda(preprocess_input)...")
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input_registered})
# print("✅ Model loaded successfully.")

# with open(CLASS_JSON, 'r', encoding='utf-8') as f:
#     class_names = json.load(f)
# print(f"Loaded {len(class_names)} class labels.")

# # ---------------------------
# # Image preprocessing function (real-world robust)
# # ---------------------------
# def load_and_preprocess_real_image(img_path):
#     img = Image.open(img_path).convert("RGB")
#     # Normalize orientation & pad to square
#     img = ImageOps.exif_transpose(img)
#     img = ImageOps.fit(img, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
#     arr = np.array(img).astype(np.float32)
#     arr = preprocess_input(arr)       # ensure same normalization as training
#     arr = np.expand_dims(arr, axis=0)
#     return arr

# # ---------------------------
# # Prediction function
# # ---------------------------
# def predict_image(img_path):
#     arr = load_and_preprocess_real_image(img_path)
#     preds = model.predict(arr, verbose=0)[0]
#     idx = np.argmax(preds)
#     conf = preds[idx] * 100
#     return class_names[idx], conf

# # ---------------------------
# # Run predictions
# # ---------------------------
# rows = []
# img_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# if not img_files:
#     raise SystemExit("❌ No images found in test directory!")

# print(f"Predicting on {len(img_files)} real-world test images...")

# for i, file in enumerate(sorted(img_files)):
#     path = os.path.join(TEST_DIR, file)
#     label, conf = predict_image(path)
#     rows.append([file, label, f"{conf:.2f}%"])
#     print(f"{file}: {label} ({conf:.2f}%)")

# # Save results to CSV
# with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Filename", "PredictedLabel", "Confidence(%)"])
#     writer.writerows(rows)

# print("✅ Predictions saved to:", OUT_CSV)

# # ---------------------------
# # Optional: Confusion Matrix & Report (if labeled subfolders exist)
# # ---------------------------
# if os.path.isdir(TEST_DIR) and any(os.path.isdir(os.path.join(TEST_DIR, d)) for d in os.listdir(TEST_DIR)):
#     print("\n📊 Evaluating confusion matrix and classification report...")
#     true_labels, pred_labels = [], []
#     for class_dir in sorted(os.listdir(TEST_DIR)):
#         class_path = os.path.join(TEST_DIR, class_dir)
#         if not os.path.isdir(class_path):
#             continue
#         for file in os.listdir(class_path):
#             if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 continue
#             fpath = os.path.join(class_path, file)
#             pred, conf = predict_image(fpath)
#             true_labels.append(class_dir)
#             pred_labels.append(pred)

#     cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
#     plt.figure(figsize=(14, 12))
#     sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix - Real World Test Set")
#     plt.tight_layout()
#     plt.savefig(os.path.join(BASE_DIR, "confusion_matrix_realworld_fixed.png"))
#     plt.close()
#     print("✅ Confusion matrix saved.")

#     report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
#     with open(os.path.join(BASE_DIR, "classification_report_realworld.txt"), "w", encoding="utf-8") as f:
#         f.write(report)
#     print("✅ Classification report saved.")

# else:
#     print("\n(No labeled subfolders found — skipped confusion matrix.)")


import os
import json
import numpy as np
import csv
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

# ---------------------------------------------------
# Register preprocess_input (for Lambda layer)
# ---------------------------------------------------
@register_keras_serializable(package="EfficientNet")
def preprocess_input_registered(x):
    return preprocess_input(x)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = r"C:\Youtube\DiseaseDetection"
MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model_transfer.keras")
CLASS_JSON = os.path.join(BASE_DIR, "class_names_transfer.json")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "real_world_test")
OUT_CSV = os.path.join(BASE_DIR, "real_test_predictions_transfer.csv")

# ---------------------------------------------------
# Load model and classes
# ---------------------------------------------------
print("Loading model with registered Lambda(preprocess_input)...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input_registered})
print("✅ Model loaded successfully.")

with open(CLASS_JSON, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print(f"Loaded {len(class_names)} class labels.")

# ---------------------------------------------------
# Real-world image preprocessing
# ---------------------------------------------------
def preprocess_real_image(img_path, image_size=(180, 180)):
    img = Image.open(img_path).convert("RGB")

    # --- Color and contrast correction ---
    img = ImageEnhance.Color(img).enhance(1.2)      # boost colors slightly
    img = ImageEnhance.Contrast(img).enhance(1.3)   # improve contrast
    img = ImageEnhance.Brightness(img).enhance(1.1) # brighten a little

    # --- Resize ---
    img = img.resize(image_size)

    # --- Convert to array ---
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # --- EfficientNet normalization ---
    arr = preprocess_input(arr)
    return arr

# ---------------------------------------------------
# Predict function with Test-Time Augmentation (TTA)
# ---------------------------------------------------
def predict_with_tta(img_path, n_augment=3):
    preds = []
    for _ in range(n_augment):
        arr = preprocess_real_image(img_path)
        pred = model.predict(arr, verbose=0)[0]
        preds.append(pred)
    mean_pred = np.mean(preds, axis=0)
    idx = np.argmax(mean_pred)
    conf = mean_pred[idx] * 100
    return class_names[idx], conf

# ---------------------------------------------------
# Run predictions
# ---------------------------------------------------
img_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not img_files:
    raise SystemExit("❌ No images found in test directory!")

print(f"Predicting on {len(img_files)} real-world test images...")

rows = []
for i, file in enumerate(sorted(img_files)):
    path = os.path.join(TEST_DIR, file)
    label, conf = predict_with_tta(path)
    rows.append([file, label, f"{conf:.2f}%"])
    print(f"{file}: {label} ({conf:.2f}%)")

# Save to CSV
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "PredictedLabel", "Confidence(%)"])
    writer.writerows(rows)

print("✅ Predictions saved to:", OUT_CSV)
print("🎯 Enhanced real-world inference complete!")
