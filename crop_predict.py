import joblib
import numpy as np
import pandas as pd

# --- Feature names used during training ---
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def main():
    # Load trained model and label encoder
    model = joblib.load("best_crop_pipeline.joblib")
    le = joblib.load("label_encoder.joblib")

    print("\n🌾 Crop Recommendation System 🌾")
    print("Enter soil and environmental details below:\n")

    # Get user inputs
    inputs = {}
    for feat in FEATURES:
        while True:
            try:
                val = float(input(f"Enter {feat}: "))
                inputs[feat] = val
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    # Convert to DataFrame for prediction
    X_in = pd.DataFrame([inputs])[FEATURES]

    # Predict probabilities
    if hasattr(model.named_steps['clf'], "predict_proba"):
        probs = model.predict_proba(X_in)[0]
        topk_idx = np.argsort(probs)[::-1][:3]
        topk = [(le.classes_[i], probs[i]) for i in topk_idx]
    else:
        pred = model.predict(X_in)[0]
        topk = [(le.inverse_transform([pred])[0], 1.0)]

    print("\n✅ Top 3 Recommended Crops:")
    for i, (label, prob) in enumerate(topk, 1):
        print(f"  {i}. {label.capitalize()} — {prob*100:.2f}%")

    print("\nRaw input data:", inputs)
    print("\n--- Prediction complete ---")

if __name__ == "__main__":
    main()

