from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
# ✨ SECURITY: Import functions for securely hashing and checking passwords
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
import os

# --- File Paths ---
# Get the absolute path of the directory where this script is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✨ DEPLOYMENT: Define explicit paths for templates and static files.
# This ensures Flask knows where to find your HTML, CSS, and JS files.
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# --- Initialize Flask ---
# Explicitly tell Flask where to find the template and static files.
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ⚠️ SECURITY: Always change this to a long, random, and secret string for production!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', "a-very-long-and-random-secret-key-should-go-here")


# --- Model and Data Paths ---
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
        pd.DataFrame(columns=['username', 'email', 'password_hash']).to_csv(USERS_PATH, index=False)
    return pd.read_csv(USERS_PATH)

def save_users_df(df):
    """Saves the DataFrame to the users CSV."""
    df.to_csv(USERS_PATH, index=False)


# --- AUTH ROUTES ---
@app.route('/')
def root():
    """
    If a user is in the session, go to the main app.
    Otherwise, redirect to the login page.
    """
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

        if users_df['username'].eq(username).any() or users_df['email'].eq(email).any():
            flash("❌ Username or email already exists. Please try another.", "danger")
            return redirect(url_for('signup'))

        password_hash = generate_password_hash(password)
        new_user = pd.DataFrame({
            'username': [username], 'email': [email], 'password_hash': [password_hash]
        })
        updated_users_df = pd.concat([users_df, new_user], ignore_index=True)
        save_users_df(updated_users_df)
        flash("✅ Signup successful! You can now log in.", "success")
        return redirect(url_for('login'))
    # Renders the single index.html but tells it to show the signup form
    return render_template('index.html', show_signup=True)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users_df = get_users_df()
        user_record = users_df[users_df['username'] == username]

        if not user_record.empty:
            stored_hash = user_record.iloc[0]['password_hash']
            if check_password_hash(stored_hash, password):
                session['user'] = username
                flash(f"✅ Welcome back, {username}!", "success")
                return redirect(url_for('home'))

        flash("❌ Invalid username or password. Please try again.", "danger")
        return redirect(url_for('login'))
    # Renders the single index.html but tells it to show the login form
    return render_template('index.html', show_login=True)


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("✅ You have been successfully logged out.", "info")
    return redirect(url_for('login'))


# --- MAIN APPLICATION ROUTE ---
@app.route('/home')
def home():
    """
    This is the main protected route for the symptom analyzer.
    """
    if "user" not in session:
        flash("🔒 You must be logged in to view that page.", "warning")
        return redirect(url_for('login'))
    
    # Renders index.html, shows the main app, and passes symptom data.
    return render_template('index.html', show_main_app=True, symptoms=symptoms, user=session['user'])


# --- PREDICTION API ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    if "user" not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    
    if model is None or medications_df is None or not symptoms:
        return jsonify({'error': 'Server not configured properly. Please contact support.'}), 500

    selected_symptoms = request.form.getlist('symptoms')
    if not selected_symptoms:
        return jsonify({'error': 'Please select at least one symptom to analyze.'}), 400

    try:
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_data[symptoms.index(symptom)] = 1

        prediction = model.predict(input_data.reshape(1, -1))[0]
        prediction_proba = model.predict_proba(input_data.reshape(1, -1))
        confidence = round(np.max(prediction_proba) * 100, 2)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

