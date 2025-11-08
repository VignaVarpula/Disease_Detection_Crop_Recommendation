# # predict.py
# """
# Batch-predict new images in a folder and save results to CSV.
# Default uses 'best_model.keras' in the same folder (recommended).
# Usage:
#   python predict.py --images_dir "C:/Youtube/DiseaseDetection/real_world_test" --model "C:/Youtube/DiseaseDetection/best_model.keras"
# """

# import os
# import argparse
# import csv
# import json
# import numpy as np
# import tensorflow as tf
# import cv2

# def load_class_names(model_path):
#     possible = [
#         os.path.join(os.path.dirname(model_path), 'class_names.json'),
#         os.path.join(model_path, 'class_names.json'),
#         'class_names.json'
#     ]
#     for p in possible:
#         if os.path.exists(p):
#             with open(p, 'r') as f:
#                 return json.load(f)
#     return None

# def preprocess_image_cv2(img_path, img_size=(128,128)):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Cannot read image: {img_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
#     img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
#     img = cv2.GaussianBlur(img, (3,3), 0)
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#     img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
#     img = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
#     arr = img.astype('float32')
#     arr = np.expand_dims(arr, axis=0)
#     return arr

# def main(args):
#     model_path = args.model
#     images_dir = args.images_dir
#     output_csv = os.path.join(images_dir, args.output_csv)

#     if not os.path.exists(model_path):
#         raise SystemExit(f"Model not found: {model_path}")
#     if not os.path.isdir(images_dir):
#         raise SystemExit(f"Images folder not found: {images_dir}")

#     class_names = load_class_names(model_path)
#     if class_names is None:
#         print("Warning: class_names.json not found. Predictions will still run but label names may be missing.")

#     model = tf.keras.models.load_model(model_path)
#     print(f"Loaded model: {model_path}")

#     image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
#     print(f"Found {len(image_files)} images in {images_dir}")

#     results = []
#     for img_name in image_files:
#         full_path = os.path.join(images_dir, img_name)
#         try:
#             arr = preprocess_image_cv2(full_path, img_size=(128,128))
#             preds = model.predict(arr, verbose=0)[0]
#             top_idx = int(np.argmax(preds))
#             conf = float(preds[top_idx])
#             if class_names:
#                 label = class_names[top_idx]
#             else:
#                 label = str(top_idx)
#             if conf < args.confidence_threshold:
#                 label = "Uncertain / Unknown"
#             results.append((img_name, label, f"{conf*100:.2f}"))
#             print(f"{img_name:<40} -> {label} ({conf*100:.2f}%)")
#         except Exception as e:
#             print(f"Error processing {img_name}: {e}")
#             results.append((img_name, "ERROR", "0.00"))

#     # write CSV
#     with open(output_csv, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['image', 'predicted_label', 'confidence_percent'])
#         writer.writerows(results)

#     print(f"\nPredictions saved to: {output_csv}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--images_dir', type=str, required=True, help='Folder with images to predict')
#     parser.add_argument('--model', type=str, default='best_model_real_image.keras', help='Path to Keras model file')
#     parser.add_argument('--output_csv', type=str, default='predictions_new_real_image.csv', help='Output CSV filename inside images_dir')
#     parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Min confidence to accept prediction (0-1). Lower to accept more predictions.')
#     args = parser.parse_args()
#     main(args)


# predict.py
"""
Batch predict plant disease on real-world images using trained CNN.
Outputs CSV with predicted class and confidence.
"""

import tensorflow as tf
import numpy as np
import os, csv, json, cv2

MODEL_PATH = r"C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras"
IMAGES_DIR = r"C:\Youtube\DiseaseDetection\dataset\real_world_test"
OUTPUT_CSV = os.path.join(IMAGES_DIR, "predictions_realworld.csv")

# Load class names
with open(r"C:\Youtube\DiseaseDetection\class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(img_path, img_size=(128,128)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    # mild cleaning
    img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=8)
    img = np.expand_dims(img, axis=0).astype("float32")
    return img

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}\n")

    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = [["Image_Name", "Predicted_Label", "Confidence (%)"]]

    for img_file in images:
        img_path = os.path.join(IMAGES_DIR, img_file)
        try:
            arr = preprocess_image(img_path)
            preds = model.predict(arr, verbose=0)[0]
            top_idx = np.argmax(preds)
            conf = np.max(preds) * 100
            label = class_names[top_idx] if conf >= 50 else "Uncertain / Unknown"
            results.append([img_file, label, f"{conf:.2f}"])
            print(f"{img_file:<40} → {label} ({conf:.2f}%)")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(results)
    print(f"\n✅ Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
