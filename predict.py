#predict.py
# #-------------------------------------------------------for epoch 6 and 10 no real world images only train ....-------------------------------------------------------------------------------------
import tensorflow as tf
import os
import csv
import numpy as np

# Path to your saved model
MODEL_PATH = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model_real_image.keras'

NEW_IMAGES_FOLDER = r'C:\Youtube\DiseaseDetection\dataset\test_images'


# Your class names (38 classes)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']   

def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    return img

def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    # Collect image paths
    image_paths = [os.path.join(NEW_IMAGES_FOLDER, f) for f in os.listdir(NEW_IMAGES_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_paths)} images for prediction.")

    # Prepare list to save predictions
    results = []

    for img_path in image_paths:
        img = load_and_preprocess_image(img_path)
        img = tf.expand_dims(img, axis=0)  # add batch dimension

        pred = model.predict(img)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_label = class_names[pred_class]

        print(f"{os.path.basename(img_path)} => Predicted: {pred_label}")
        results.append([os.path.basename(img_path), pred_label])

    # Save results to CSV
    csv_path = os.path.join(NEW_IMAGES_FOLDER, 'real_predictions_test1_real_images.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted_Class'])
        writer.writerows(results)

    print(f"Predictions saved to {csv_path}")

if __name__ == "__main__":
    main()

# import tensorflow as tf
# import numpy as np
# import os
# import csv
# from PIL import Image

# # Paths
# MODEL_PATH = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras'
# NEW_IMAGES_FOLDER = r'C:\Youtube\DiseaseDetection\dataset\test'  # Your new images folder
# CSV_SAVE_PATH = os.path.join(NEW_IMAGES_FOLDER, 'predictions.csv')

# # Class names must match training order exactly
# class_names = [
#     "Apple___Apple_scab",
#     "Apple___Black_rot",
#     "Apple___Cedar_apple_rust",
#     "Apple___healthy",
#     "Blueberry___healthy",
#     "Cherry_(including_sour)_Powdery_mildew",
#     "Cherry_(including_sour)_healthy",
#     "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
#     "Corn_(maize)Common_rust",
#     "Corn_(maize)_Northern_Leaf_Blight",
#     "Corn_(maize)_healthy",
#     "Grape___Black_rot",
#     "Grape__Esca(Black_Measles)",
#     "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
#     "Grape___healthy",
#     "Orange__Haunglongbing(Citrus_greening)",
#     "Peach___Bacterial_spot",
#     "Peach___healthy",
#     "Pepper,bell__Bacterial_spot",
#     "Pepper,bell__healthy",
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Raspberry___healthy",
#     "Soybean___healthy",
#     "Squash___Powdery_mildew",
#     "Strawberry___Leaf_scorch",
#     "Strawberry___healthy",
#     "Tomato___Bacterial_spot",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot",
#     "Tomato___Spider_mites Two-spotted_spider_mite",
#     "Tomato___Target_Spot",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "Tomato___Tomato_mosaic_virus",
#     "Tomato___healthy"
# ]

# def preprocess_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize((128, 128))
#     img = np.array(img) / 255.0  # Normalize to [0,1]
#     img = np.expand_dims(img, axis=0)  # Batch dimension
#     return img

# def main():
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print(f"Model loaded from {MODEL_PATH}")

#     image_paths = [os.path.join(NEW_IMAGES_FOLDER, f) for f in os.listdir(NEW_IMAGES_FOLDER)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

#     print(f"Found {len(image_paths)} images to predict.")

#     results = []

#     for img_path in image_paths:
#         img = preprocess_image(img_path)
#         preds = model.predict(img)
#         pred_index = np.argmax(preds)
#         pred_label = class_names[pred_index]
#         print(f"{os.path.basename(img_path)} => Predicted: {pred_label}")
#         results.append([os.path.basename(img_path), pred_label])

#     # Save to CSV
#     with open(CSV_SAVE_PATH, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Image', 'Predicted_Class'])
#         writer.writerows(results)

#     print(f"Predictions saved to {CSV_SAVE_PATH}")

# if __name__ == "__main__":
#     main()

# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import csv

# # =========================
# # Paths
# # =========================
# MODEL_PATH = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras'
# IMAGES_FOLDER = r'C:\Youtube\DiseaseDetection\dataset\test'
# OUTPUT_CSV = os.path.join(IMAGES_FOLDER, 'predictions.csv')

# # =========================
# # Class Names
# # =========================
# class_names = [
#     "Apple___Apple_scab",
#     "Apple___Black_rot",
#     "Apple___Cedar_apple_rust",
#     "Apple___healthy",
#     "Blueberry___healthy",
#     "Cherry_(including_sour)___Powdery_mildew",
#     "Cherry_(including_sour)___healthy",
#     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
#     "Corn_(maize)___Common_rust_",
#     "Corn_(maize)___Northern_Leaf_Blight",
#     "Corn_(maize)___healthy",
#     "Grape___Black_rot",
#     "Grape___Esca_(Black_Measles)",
#     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
#     "Grape___healthy",
#     "Orange___Haunglongbing_(Citrus_greening)",
#     "Peach___Bacterial_spot",
#     "Peach___healthy",
#     "Pepper,_bell___Bacterial_spot",
#     "Pepper,_bell___healthy",
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Raspberry___healthy",
#     "Soybean___healthy",
#     "Squash___Powdery_mildew",
#     "Strawberry___Leaf_scorch",
#     "Strawberry___healthy",
#     "Tomato___Bacterial_spot",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot",
#     "Tomato___Spider_mites Two-spotted_spider_mite",
#     "Tomato___Target_Spot",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "Tomato___Tomato_mosaic_virus",
#     "Tomato___healthy"
# ]

# # =========================
# # Preprocess Image
# # =========================
# def preprocess_image_cv2(img_path, img_size=(128, 128)):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"❌ Cannot read image: {img_path}")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, img_size)

#     # ✅ Optional: Light color normalization for real-world images
#     img = cv2.bilateralFilter(img, 5, 75, 75)
#     img = cv2.GaussianBlur(img, (3, 3), 0)

#     # ❌ Don't divide by 255 here — model already rescales internally
#     img = np.expand_dims(img, axis=0)
#     return img

# # =========================
# # Prediction Script
# # =========================
# def main():
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print(f"✅ Model loaded: {MODEL_PATH}\n")

#     # Collect images
#     image_paths = [
#         os.path.join(IMAGES_FOLDER, f)
#         for f in os.listdir(IMAGES_FOLDER)
#         if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#     ]

#     if not image_paths:
#         print("❌ No image files found!")
#         return

#     print(f"Found {len(image_paths)} images for prediction.\n")

#     results = [["Image_Name", "Predicted_Class", "Confidence (%)"]]

#     for path in image_paths:
#         try:
#             img = preprocess_image_cv2(path)
#             preds = model.predict(img, verbose=0)
#             pred_class = np.argmax(preds)
#             pred_label = class_names[pred_class]
#             confidence = np.max(preds) * 100

#             if confidence < 50:
#                 pred_label = "Uncertain / Unknown"

#             print(f"{os.path.basename(path):<35} → {pred_label} ({confidence:.2f}%)")
#             results.append([os.path.basename(path), pred_label, f"{confidence:.2f}"])
#         except Exception as e:
#             print(f"Error: {path} — {e}")

#     # Save results
#     with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
#         csv.writer(f).writerows(results)

#     print(f"\n✅ Predictions saved to: {OUTPUT_CSV}\n")

# # =========================
# # Run
# # =========================
# if __name__ == "__main__":
#     main()

