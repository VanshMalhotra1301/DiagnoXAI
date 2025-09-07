from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle
import pandas as pd
import numpy as np
import os


# --- Initialize Flask ---
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # needed for session handling


# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_predictor.pkl')
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_PATH = os.path.join(DATA_DIR, 'users.csv')

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
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop('Unnamed: 133', axis=1)
    symptoms = train_df.drop('prognosis', axis=1).columns.tolist()
    print(f"✅ Symptoms list loaded successfully with {len(symptoms)} symptoms.")
except Exception as e:
    print(f"❌ ERROR loading Training.csv for symptom list: {e}")
    symptoms = []
print("--- Initialization Complete ---")


# --- AUTH ROUTES ---
@app.route('/')
def root():
    """Default route → always go to login"""
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not os.path.exists(USERS_PATH):
            pd.DataFrame(columns=['username', 'email', 'password']).to_csv(USERS_PATH, index=False)

        users_df = pd.read_csv(USERS_PATH)

        if ((users_df['username'] == username).any()) or ((users_df['email'] == email).any()):
            flash("❌ Username or email already exists. Try again.", "danger")
            return redirect(url_for('signup'))

        new_user = pd.DataFrame([[username, email, password]], columns=['username', 'email', 'password'])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_PATH, index=False)

        flash("✅ Signup successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not os.path.exists(USERS_PATH):
            flash("❌ No users registered yet. Please signup first.", "danger")
            return redirect(url_for('signup'))

        users_df = pd.read_csv(USERS_PATH)

        user = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
        if not user.empty:
            session['user'] = username
            flash(f"✅ Welcome, {username}!", "success")
            return redirect(url_for('home'))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("✅ Logged out successfully.", "success")
    return redirect(url_for('login'))


# --- PROTECTED HOME ROUTE ---
@app.route('/home')
def home():
    if "user" not in session:
        return redirect(url_for('login'))
    return render_template('index.html', symptoms=symptoms, user=session['user'])


# --- PREDICTION ROUTE ---
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


# --- Run Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
