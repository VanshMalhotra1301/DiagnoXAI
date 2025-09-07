import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- Starting Model Training ---")
try:
    train_df = pd.read_csv('Training.csv')
    test_df = pd.read_csv('Testing.csv')
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    exit()

# Drop the 'Unnamed: 133' column if it exists in the training data
if 'Unnamed: 133' in train_df.columns:
    train_df = train_df.drop('Unnamed: 133', axis=1)
    print("✅ Dropped 'Unnamed: 133' column from training data.")

# Prepare the data
X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

print(f"Training model with {len(X_train.columns)} features.")

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Model training complete!")

# --- THIS IS THE MISSING LINE ---
# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test)

# Now, evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save the final model
if not os.path.exists('models'):
    os.makedirs('models')
with open('models/disease_predictor.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("🚀 New, corrected model saved successfully to models/disease_predictor.pkl")