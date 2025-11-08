# ---------------------------------------Copilot VS Version--------------------------
# # train_disease_transfer.py
# import os
# import json
# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import csv

# import tensorflow as tf
# from tensorflow.keras import layers, models # type: ignore
# from tensorflow.keras.applications import EfficientNetB0 # type: ignore
# from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
# from sklearn.metrics import confusion_matrix, classification_report

# # -------------------------
# # Configuration / paths
# # -------------------------
# TRAIN_DIR = r'C:\Youtube\DiseaseDetection\dataset\train'
# VALID_DIR = r'C:\Youtube\DiseaseDetection\dataset\valid'
# TEST_DIR = r'C:\Youtube\DiseaseDetection\dataset\test_images'
# BEST_MODEL_PATH = r'C:\Youtube\DiseaseDetection\best_model_transfer.keras'
# FINAL_MODEL_PATH = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model_transfer.keras'
# CLASS_JSON = r'C:\Youtube\DiseaseDetection\class_names_transfer.json'
# HISTORY_PATH = r'C:\Youtube\DiseaseDetection\training_hist_transfer.json'
# CONF_MATRIX_PNG = r'C:\Youtube\DiseaseDetection\confusion_matrix_transfer.png'
# CLASS_REPORT_TXT = r'C:\Youtube\DiseaseDetection\classification_report_transfer.txt'

# # Smoke-test / lower memory settings
# IMAGE_SIZE = (128, 128)   # smaller for quick runs; increase later if you have RAM/GPU
# BATCH_SIZE = 8

# # dataset performance tuning
# AUTOTUNE = tf.data.AUTOTUNE
# # Use smaller epoch counts for a quick smoke test; increase for real training
# NUM_HEAD_EPOCHS = 1  # quick head training for smoke test
# NUM_FINE_TUNE_EPOCHS = 0  # skip fine-tuning in smoke test; increase when ready
# LEARNING_RATE_HEAD = 1e-4
# LEARNING_RATE_FINE = 1e-5

# # -------------------------
# # Dataset creation
# # -------------------------
# print("Creating datasets...")
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     TRAIN_DIR,
#     label_mode='int',   # we will use sparse_categorical_crossentropy and class_weight
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     VALID_DIR,
#     label_mode='int',
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# class_names = train_ds.class_names
# num_classes = len(class_names)
# print(f"Found {num_classes} classes.")

# # Save class names for prediction script
# with open(CLASS_JSON, 'w', encoding='utf-8') as f:
#     json.dump(class_names, f, ensure_ascii=False, indent=2)

# # Improve pipeline performance (avoid caching whole dataset in RAM)
# train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# # Create test dataset if available (supports labeled subfolders or unlabeled images)
# test_ds = None
# test_unlabeled_paths = None
# if os.path.isdir(TEST_DIR):
#     subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
#     if subdirs:
#         test_ds = tf.keras.utils.image_dataset_from_directory(
#             TEST_DIR,
#             label_mode='int',
#             image_size=IMAGE_SIZE,
#             batch_size=BATCH_SIZE,
#             shuffle=False
#         )
#         test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
#         print("Labeled test dataset created from:", TEST_DIR)
#     else:
#         image_paths = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
#                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]
#         if image_paths:
#             test_unlabeled_paths = sorted(image_paths)
#             print(f"Found {len(test_unlabeled_paths)} unlabeled test images in: {TEST_DIR}")
#         else:
#             print("No images found in test directory:", TEST_DIR)
# else:
#     print("No test dataset folder found:", TEST_DIR)

# # Compute class weights from folder counts (fallback if dataset is large)
# def compute_class_weights_from_dir(train_dir):
#     counts = {}
#     for class_name in os.listdir(train_dir):
#         class_path = os.path.join(train_dir, class_name)
#         if os.path.isdir(class_path):
#             n = sum(1 for _ in os.listdir(class_path) if _.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
#             counts[class_name] = n
#     # map to index weights
#     total = sum(counts.values())
#     weights = {}
#     for idx, name in enumerate(sorted(counts.keys())):
#         count = counts.get(name, 0)
#         if count == 0:
#             weights[idx] = 1.0
#         else:
#             weights[idx] = total / (len(counts) * count)
#     return weights

# class_weight = compute_class_weights_from_dir(TRAIN_DIR)
# print("Class weights (sample):", dict(list(class_weight.items())[:5]))

# # -------------------------
# # Data augmentation
# # -------------------------
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.12),
#     layers.RandomZoom(0.08),
#     layers.RandomContrast(0.08),
# ], name="data_augmentation")

# # -------------------------
# # Model building (transfer learning)
# # -------------------------
# print("Building model (EfficientNetB0 base)...")
# base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
# base_model.trainable = False  # freeze for head training

# inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
# x = data_augmentation(inputs)
# # EfficientNet expects preprocessing: use preprocess_input
# x = layers.Lambda(preprocess_input, name="preprocess")(x)
# x = base_model(x, training=False)
# x = layers.GlobalAveragePooling2D(name="gap")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

# model = models.Model(inputs, outputs, name="EffNetB0_transfer")
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # -------------------------
# # Callbacks
# # -------------------------
# callbacks = [
#     ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
#     EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
# ]

# # -------------------------
# # Train head (frozen base)
# # -------------------------
# print(f"Training head for {NUM_HEAD_EPOCHS} epochs...")
# history_head = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=NUM_HEAD_EPOCHS,
#     callbacks=callbacks,
#     class_weight=class_weight
# )

# # -------------------------
# # Fine-tune (unfreeze some top layers)
# # -------------------------
# print("Unfreezing top layers for fine-tuning...")
# base_model.trainable = True
# # Freeze all layers except the last N layers (tune N as needed)
# fine_tune_at = len(base_model.layers) - 40  # unfreeze last 40 layers
# for i, layer in enumerate(base_model.layers):
#     layer.trainable = i >= fine_tune_at

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# print("Fine-tuning model...")
# history_fine = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=NUM_HEAD_EPOCHS + NUM_FINE_TUNE_EPOCHS,
#     initial_epoch=history_head.epoch[-1] + 1 if history_head.epoch else 0,
#     callbacks=callbacks,
#     class_weight=class_weight
# )

# # -------------------------
# # Save final model & history
# # -------------------------
# print("Saving final model and history...")
# model.save(FINAL_MODEL_PATH)

# # Combine histories
# history = {}
# for k in history_head.history:
#     history[k] = history_head.history[k]
# for k, v in history_fine.history.items():
#     history.setdefault(k, []).extend(v)

# os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
# with open(HISTORY_PATH, 'w') as f:
#     json.dump(history, f, indent=2)

# print("Training complete. Model saved to:", FINAL_MODEL_PATH)

# # =========================
# # Evaluation on test set (confusion matrix + classification report)
# # =========================
# if test_ds is not None:
#     print("Evaluating on test set...")
#     test_loss, test_acc = model.evaluate(test_ds, verbose=1)
#     print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

#     # Get predictions and true labels
#     print("Collecting predictions on test set (this may take a while)...")
#     preds = model.predict(test_ds, verbose=1)
#     y_pred = np.argmax(preds, axis=1)
#     # gather true labels from dataset
#     y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

#     # Confusion matrix and classification report
#     cm = confusion_matrix(y_true, y_pred)
#     report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

#     # Save classification report
#     with open(CLASS_REPORT_TXT, 'w', encoding='utf-8') as f:
#         f.write(f"Test Loss: {test_loss:.6f}\nTest Accuracy: {test_acc:.6f}\n\n")
#         f.write(report)
#     print("Classification report saved to:", CLASS_REPORT_TXT)

#     # Plot confusion matrix heatmap
#     plt.figure(figsize=(16, 14))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix - Test Set')
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(CONF_MATRIX_PNG)
#     plt.close()
#     print("Confusion matrix image saved to:", CONF_MATRIX_PNG)

# else:
#     print("Skipping test evaluation because test dataset not found.")
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# -------------------------
# Configuration / paths
# -------------------------
TRAIN_DIR = r"C:\Youtube\DiseaseDetection\dataset\train"
VALID_DIR = r"C:\Youtube\DiseaseDetection\dataset\valid"
TEST_DIR = r"C:\Youtube\DiseaseDetection\dataset\test_images"

BEST_MODEL_PATH = r"C:\Youtube\DiseaseDetection\best_model_transfer.keras"
FINAL_MODEL_PATH = r"C:\Youtube\DiseaseDetection\trained_plant_disease_model_transfer.keras"
CLASS_JSON = r"C:\Youtube\DiseaseDetection\3.json"
HISTORY_PATH = r"C:\Youtube\DiseaseDetection\training_hist_transfer.json"
CONF_MATRIX_PNG = r"C:\Youtube\DiseaseDetection\confusion_matrix_transfer.png"
CLASS_REPORT_TXT = r"C:\Youtube\DiseaseDetection\classification_report_transfer.txt"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 8
NUM_HEAD_EPOCHS = 6
NUM_FINE_TUNE_EPOCHS = 6
LEARNING_RATE_HEAD = 1e-4u
LEARNING_RATE_FINE = 1e-5

# -------------------------
# Force CPU mode
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("✅ Training on CPU only (GPU disabled).")

# -------------------------
# Dataset creation
# -------------------------
print("📂 Loading train/validation datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, label_mode='int', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, label_mode='int', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes.")

# Save class names for prediction use
with open(CLASS_JSON, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------
# Data augmentation
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
], name="data_augmentation")

# -------------------------
# Build model
# -------------------------
print("⚙️ Building EfficientNetB0 model...")
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# Callbacks
# -------------------------
callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

# -------------------------
# Training (head + fine-tune)
# -------------------------
print(f"🚀 Training top layers for {NUM_HEAD_EPOCHS} epochs...")
history_head = model.fit(train_ds, validation_data=val_ds, epochs=NUM_HEAD_EPOCHS, callbacks=callbacks)

print("🔧 Fine-tuning last layers...")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40
for i, layer in enumerate(base_model.layers):
    layer.trainable = i >= fine_tune_at

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_HEAD_EPOCHS + NUM_FINE_TUNE_EPOCHS,
    initial_epoch=history_head.epoch[-1] + 1,
    callbacks=callbacks
)

# -------------------------
# Save model & history
# -------------------------
print("💾 Saving model and training history...")
model.save(FINAL_MODEL_PATH)
history = {k: history_head.history.get(k, []) + history_fine.history.get(k, []) for k in set(history_head.history) | set(history_fine.history)}
with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, indent=2)

# -------------------------
# Evaluate / Predict on Test Dataset
# -------------------------
if os.path.isdir(TEST_DIR):
    subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    if subdirs:
        print("📊 Found labeled test dataset, computing confusion matrix...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR, label_mode='int', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
        )
        preds = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

        with open(CLASS_REPORT_TXT, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Classification report saved to {CLASS_REPORT_TXT}")

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Test Set")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PNG)
        plt.close()
        print(f"✅ Confusion matrix saved to {CONF_MATRIX_PNG}")

    else:
        print("🧪 Found unlabeled test images. Running predictions...")
        test_images = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        preds = []
        for img_path in test_images:
            img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = model.predict(arr, verbose=0)
            label = class_names[np.argmax(pred)]
            conf = np.max(pred) * 100
            preds.append((os.path.basename(img_path), label, f"{conf:.2f}%"))

        csv_path = os.path.join(os.path.dirname(TEST_DIR), "test_predictions.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Filename", "PredictedLabel", "Confidence(%)"])
            writer.writerows(preds)
        print(f"✅ Predictions saved to {csv_path}")

else:
    print("⚠️ Test folder not found; skipping test evaluation.")

print("🎯 Training complete! Model ready for real-world prediction.")
