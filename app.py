from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_predictor.pkl')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Load Model and Data ---
print("--- Initializing Application ---")
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    model = None

try:
    medications_df = pd.read_csv(os.path.join(DATA_DIR, 'medications.csv'))
    print("✅ Medications data loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading medications.csv: {e}")
    medications_df = None

try:
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'Training.csv'))
    
    # --- THE SAME CORRECT FIX IS HERE ---
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop('Unnamed: 133', axis=1)
        
    symptoms = train_df.drop('prognosis', axis=1).columns.tolist()
    print(f"✅ Symptoms list loaded successfully with {len(symptoms)} symptoms.")
except Exception as e:
    print(f"❌ ERROR loading Training.csv for symptom list: {e}")
    symptoms = []
print("--- Initialization Complete ---")

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or medications_df is None or not symptoms:
        return jsonify({'error': 'Server not configured properly.'}), 500

    selected_symptoms = request.form.getlist('symptoms')
    if not selected_symptoms:
        return jsonify({'error': 'Please select at least one symptom.'}), 400

    try:
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_data[symptoms.index(symptom)] = 1
        
        input_data = input_data.reshape(1, -1)
        prediction = model.predict(input_data)[0]
        
        suggestion_row = medications_df[medications_df['Disease'].str.lower() == prediction.lower()]
        suggestion = suggestion_row['Suggestion'].iloc[0] if not suggestion_row.empty else "Consult a doctor for advice."

        return jsonify({'prediction': prediction, 'suggestion': suggestion})
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)