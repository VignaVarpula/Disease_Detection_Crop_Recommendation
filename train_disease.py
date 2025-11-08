# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# from sklearn.metrics import confusion_matrix, classification_report
# import os

# # Paths (adjust exactly for your environment)
# train_path = r'C:\Youtube\DiseaseDetection\dataset\train'
# valid_path = r'C:\Youtube\DiseaseDetection\dataset\valid'
# test_path = r'C:\Youtube\DiseaseDetection\dataset\test'

# # Load datasets
# training_set = tf.keras.utils.image_dataset_from_directory(
#     train_path,
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True
# )

# validation_set = tf.keras.utils.image_dataset_from_directory(
#     valid_path,
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True
# )

# test_set = tf.keras.utils.image_dataset_from_directory(
#     test_path,
#     label_mode="categorical",
#     batch_size=16,  # Increased batch size for faster evaluation
#     image_size=(128, 128),
#     shuffle=False
# )

# # Define model with Rescaling layer to normalize inputs [0,255] -> [0,1]
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Rescaling(1./255, input_shape=(128,128,3)),

#     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(64, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(128, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(256, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(512, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1500, activation='relu'),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(38, activation='softmax')  # 38 classes
# ])

# cnn.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# cnn.summary()

# # Callbacks: Save best model and early stopping
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=r'C:\Youtube\DiseaseDetection\best_model.keras',
#         save_best_only=True,
#         monitor='val_accuracy',
#         mode='max'
#     ),
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_accuracy',
#         patience=3,
#         restore_best_weights=True
#     )
# ]

# # Train model
# history = cnn.fit(
#     training_set,
#     validation_data=validation_set,
#     epochs=10,
#     callbacks=callbacks
# )

# # Evaluate on training and validation sets
# train_loss, train_acc = cnn.evaluate(training_set)
# print(f"Training accuracy: {train_acc}")

# val_loss, val_acc = cnn.evaluate(validation_set)
# print(f"Validation accuracy: {val_acc}")

# # Save final model
# cnn.save(r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras')

# # Save training history JSON (ensure folder exists)
# history_path = r'C:\Youtube\DiseaseDetection\training_hist.json'
# os.makedirs(os.path.dirname(history_path), exist_ok=True)
# with open(history_path, 'w') as f:
#     json.dump(history.history, f)

# # Plot training and validation accuracy
# epochs = range(1, len(history.history['accuracy']) + 1)
# plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='red')
# plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Get class names from training set (should match validation/test)
# class_names = training_set.class_names

# # Predictions on test set
# y_pred = cnn.predict(test_set)
# predicted_categories = tf.argmax(y_pred, axis=1)

# # Extract true labels (unbatch for correct concatenation)
# true_categories = tf.concat([y for x, y in test_set.unbatch()], axis=0)
# Y_true = tf.argmax(true_categories, axis=1)

# # Confusion matrix and classification report
# cm = confusion_matrix(Y_true, predicted_categories)
# print(classification_report(Y_true, predicted_categories, target_names=class_names))

# # Plot confusion matrix heatmap
# plt.figure(figsize=(20, 20))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted Class')
# plt.ylabel('Actual Class')
# plt.title('Confusion Matrix - Plant Disease Classification')
# plt.tight_layout()
# plt.show()



# #train_disease.py

# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# import os
# from sklearn.metrics import confusion_matrix, classification_report

# # Paths (adjust exactly for your environment)
# train_path = r'C:\Youtube\DiseaseDetection\dataset\train'
# valid_path = r'C:\Youtube\DiseaseDetection\dataset\valid'
# test_path = r'C:\Youtube\DiseaseDetection\dataset\test'

# # Load datasets
# training_set = tf.keras.utils.image_dataset_from_directory(
#     train_path,
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True
# )

# validation_set = tf.keras.utils.image_dataset_from_directory(
#     valid_path,
#     label_mode="categorical",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True
# )

# # For test set without subfolders, we need to handle it differently
# test_image_paths = [os.path.join(test_path, f) for f in os.listdir(test_path)
#                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

# def load_and_preprocess_image(path):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_image(img, channels=3)
#     img = tf.image.resize(img, (128, 128))
#     img = img / 255.0  # Normalize to [0,1]
#     return img

# test_images = tf.stack([load_and_preprocess_image(p) for p in test_image_paths])

# # Define model with Rescaling layer to normalize inputs [0,255] -> [0,1]
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Rescaling(1./255, input_shape=(128,128,3)),

#     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(64, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(128, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(256, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(512, 3, activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),

#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1500, activation='relu'),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(38, activation='softmax')  # 38 classes
# ])

# cnn.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# cnn.summary()

# # Callbacks: Save best model and early stopping
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=r'C:\Youtube\DiseaseDetection\best_model.keras',
#         save_best_only=True,
#         monitor='val_accuracy',
#         mode='max'
#     ),
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_accuracy',
#         patience=3,
#         restore_best_weights=True
#     )
# ]

# # Train model
# history = cnn.fit(
#     training_set,
#     validation_data=validation_set,
#     epochs=10,
#     callbacks=callbacks
# )

# # Evaluate on training and validation sets
# train_loss, train_acc = cnn.evaluate(training_set)
# print(f"Training accuracy: {train_acc}")

# val_loss, val_acc = cnn.evaluate(validation_set)
# print(f"Validation accuracy: {val_acc}")

# # Save final model
# cnn.save(r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras')

# # Save training history JSON (ensure folder exists)
# history_path = r'C:\Youtube\DiseaseDetection\training_hist.json'
# os.makedirs(os.path.dirname(history_path), exist_ok=True)
# with open(history_path, 'w') as f:
#     json.dump(history.history, f)

# # Plot training and validation accuracy
# epochs = range(1, len(history.history['accuracy']) + 1)
# plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='red')
# plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Get class names from training set (should match validation/test)
# class_names = training_set.class_names

# # Predictions on test set
# y_pred = cnn.predict(test_images)
# predicted_categories = tf.argmax(y_pred, axis=1)

# # Since test set doesn't have labels, we can't compute confusion matrix or classification report
# print("Predictions:")
# for img_path, pred in zip(test_image_paths, predicted_categories.numpy()):
#     print(f"{os.path.basename(img_path)} => Class index: {pred}")


# -------------------------------------------------------for epoch 6 and 10 no real world images only train ....-------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os

# Paths
train_path = r'C:\Youtube\DiseaseDetection\dataset\train'
valid_path = r'C:\Youtube\DiseaseDetection\dataset\valid'
best_model_path = r'C:\Youtube\DiseaseDetection\best_model.keras'
final_model_path = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras'
history_path = r'C:\Youtube\DiseaseDetection\training_hist.json'

# Load datasets
batch_size = 32
img_size = (128, 128)

training_set = tf.keras.utils.image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

# CNN Model definition
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(*img_size, 3)),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(512, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(38, activation='softmax')  # 38 classes
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()

# Callbacks: Save best weights & Early stopping
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=r'C:\Youtube\DiseaseDetection\best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
]

# Train
history = cnn.fit(
    training_set,
    validation_data=validation_set,
    epochs=10,
    callbacks=callbacks
)

# Save final model
cnn.save(final_model_path)

# Save training history
os.makedirs(os.path.dirname(history_path), exist_ok=True)
with open(history_path, 'w') as f:
    json.dump(history.history, f)

# Plot accuracy
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='red')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

print("Training complete. Model saved.")

# import tensorflow as tf
# import matplotlib.pyplot as plt
# import json, os

# # =========================
# # Paths
# # =========================
# train_path = r'C:\Youtube\DiseaseDetection\dataset\train'
# valid_path = r'C:\Youtube\DiseaseDetection\dataset\valid'
# best_model_path = r'C:\Youtube\DiseaseDetection\best_model.keras'
# final_model_path = r'C:\Youtube\DiseaseDetection\trained_plant_disease_model.keras'
# history_path = r'C:\Youtube\DiseaseDetection\training_history.json'

# # =========================
# # Dataset
# # =========================
# img_size = (128, 128)
# batch_size = 32

# train_datagen = tf.keras.utils.image_dataset_from_directory(
#     train_path,
#     label_mode='categorical',
#     batch_size=batch_size,
#     image_size=img_size,
#     shuffle=True
# )

# val_datagen = tf.keras.utils.image_dataset_from_directory(
#     valid_path,
#     label_mode='categorical',
#     batch_size=batch_size,
#     image_size=img_size,
#     shuffle=True
# )

# # =========================
# # Data Augmentation
# # =========================
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.15),
#     tf.keras.layers.RandomZoom(0.1),
#     tf.keras.layers.RandomContrast(0.2),
# ])

# # =========================
# # CNN Architecture
# # =========================
# cnn = tf.keras.models.Sequential([
#     tf.keras.layers.Rescaling(1./255, input_shape=(*img_size, 3)),
#     data_augmentation,

#     tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(38, activation='softmax')
# ])

# cnn.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# cnn.summary()

# # =========================
# # Callbacks
# # =========================
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
#     tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
# ]

# # =========================
# # Training
# # =========================
# history = cnn.fit(
#     train_datagen,
#     validation_data=val_datagen,
#     epochs=10,
#     callbacks=callbacks
# )

# # =========================
# # Save Model and History
# # =========================
# cnn.save(final_model_path)
# os.makedirs(os.path.dirname(history_path), exist_ok=True)
# with open(history_path, 'w') as f:
#     json.dump(history.history, f)

# # =========================
# # Plot Accuracy Graph
# # =========================
# plt.plot(history.history['accuracy'], label='Training Accuracy', color='red')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # =========================
# # Final Evaluation Summary
# # =========================
# train_loss, train_acc = cnnG.evaluate(train_datagen, verbose=0)
# val_loss, val_acc = cnn.evaluate(val_datagen, verbose=0)

# print("\n================ TRAINING SUMMARY ================")
# print(f"Final Training Accuracy     : {train_acc * 100:.2f}%")
# print(f"Final Validation Accuracy   : {val_acc * 100:.2f}%")

# # Load and evaluate best model for comparison
# best_model = tf.keras.models.load_model(best_model_path)
# best_val_loss, best_val_acc = best_model.evaluate(val_datagen, verbose=0)
# print(f"Best Model Validation Acc   : {best_val_acc * 100:.2f}%")
# print("===================================================\n")

# print(f"✅ Final model saved at: {final_model_path}")
# print(f"✅ Best model saved at: {best_model_path}")
# print(f"✅ Training history saved at: {history_path}")
