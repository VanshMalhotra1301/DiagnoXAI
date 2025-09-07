from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
# ✨ SECURITY: Import functions for securely hashing and checking passwords
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
import os

# --- Initialize Flask ---
app = Flask(__name__)
# ⚠️ SECURITY: Always change this to a long, random, and secret string for production!
# This key protects your user sessions from being tampered with.
app.secret_key = "a-very-long-and-random-secret-key-should-go-here"

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_predictor.pkl')
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_PATH = os.path.join(DATA_DIR, 'users.csv')
# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# --- Load Model and Data ---
print("--- Initializing Application ---")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
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
    # Clean up the common "Unnamed" column issue from CSVs
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop('Unnamed: 133', axis=1)
    symptoms = train_df.drop('prognosis', axis=1).columns.tolist()
    print(f"✅ Symptoms list loaded successfully with {len(symptoms)} symptoms.")
except Exception as e:
    print(f"❌ ERROR loading Training.csv for symptom list: {e}")
    symptoms = []
print("--- Initialization Complete ---")


# --- Helper Function to manage users.csv ---
def get_users_df():
    """Safely reads the users CSV, creating it if it doesn't exist."""
    if not os.path.exists(USERS_PATH):
        # ✨ IMPROVEMENT: Define columns for the new CSV, including the password hash
        pd.DataFrame(columns=['username', 'email', 'password_hash']).to_csv(USERS_PATH, index=False)
    return pd.read_csv(USERS_PATH)

def save_users_df(df):
    """Saves the DataFrame to the users CSV."""
    df.to_csv(USERS_PATH, index=False)


# --- AUTH ROUTES ---
@app.route('/')
def root():
    """Redirects to the login page if not logged in, otherwise to home."""
    if "user" in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        users_df = get_users_df()

        # Check if username or email already exists
        if users_df['username'].eq(username).any() or users_df['email'].eq(email).any():
            flash("❌ Username or email already exists. Please try another.", "danger")
            return redirect(url_for('signup'))

        # ✨ CRITICAL SECURITY FIX: Hash the password before storing it
        password_hash = generate_password_hash(password)

        # Create a new user record
        new_user = pd.DataFrame({
            'username': [username],
            'email': [email],
            'password_hash': [password_hash]
        })
        
        # Add the new user and save the file
        updated_users_df = pd.concat([users_df, new_user], ignore_index=True)
        save_users_df(updated_users_df)

        flash("✅ Signup successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        users_df = get_users_df()
        
        # Find the user by username
        user_record = users_df[users_df['username'] == username]

        # ✨ CRITICAL SECURITY FIX: Check the hashed password
        if not user_record.empty:
            stored_hash = user_record.iloc[0]['password_hash']
            if check_password_hash(stored_hash, password):
                session['user'] = username
                flash(f"✅ Welcome back, {username}!", "success")
                return redirect(url_for('home'))

        # If user not found or password incorrect, show the same generic error
        flash("❌ Invalid username or password. Please try again.", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("✅ You have been successfully logged out.", "info")
    return redirect(url_for('login'))


# --- PROTECTED HOME ROUTE ---
@app.route('/home')
def home():
    if "user" not in session:
        flash("🔒 You must be logged in to view that page.", "warning")
        return redirect(url_for('login'))
    # This correctly passes the list of symptoms to your index.html template
    return render_template('index.html', symptoms=symptoms, user=session['user'])


# --- PREDICTION API ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    # ✨ IMPROVEMENT: Check if user is logged in before allowing prediction
    if "user" not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    if model is None or medications_df is None or not symptoms:
        return jsonify({'error': 'Server not configured properly. Please contact support.'}), 500

    selected_symptoms = request.form.getlist('symptoms')
    if not selected_symptoms:
        return jsonify({'error': 'Please select at least one symptom to analyze.'}), 400

    try:
        # Create the input vector for the model
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_data[symptoms.index(symptom)] = 1

        # Reshape for the model and predict
        prediction = model.predict(input_data.reshape(1, -1))[0]
        prediction_proba = model.predict_proba(input_data.reshape(1, -1))
        confidence = round(np.max(prediction_proba) * 100, 2)

        # Find the corresponding suggestion from the medications dataframe
        suggestion_row = medications_df[medications_df['Disease'].str.lower() == prediction.lower()]
        suggestion = suggestion_row['Suggestion'].iloc[0] if not suggestion_row.empty else "Please consult a healthcare professional for personalized advice."

        return jsonify({
            'prediction': prediction,
            'suggestion': suggestion,
            'confidence': confidence
        })
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during analysis.'}), 500


# --- Run Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
