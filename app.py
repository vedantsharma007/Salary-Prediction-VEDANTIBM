import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import traceback

# --- 1. Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) 

# Load the trained model and feature list
try:
    model = joblib.load('salary_model.pkl')
    model_features = joblib.load('model_features.pkl')
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"An error occurred loading the model. Ensure train_model.py was run: {e}")
    sys.exit(1)


# --- Helper Function for Preprocessing (Clean and Robust) ---
def preprocess_input(data_json, features_list):
    """
    Takes raw JSON input (keys are clean), converts it to a DataFrame, 
    applies one-hot encoding, and aligns with the model's feature order.
    """
    input_df = pd.DataFrame([data_json])

    # 1. STANDARDIZE CATEGORICAL VALUES (MUST MATCH TRAINER)
    categorical_cols = ['gender', 'education_level', 'job_title'] 
    
    for col in categorical_cols:
        # Standardize the value string: 'Data Scientist' -> 'data_scientist'
        input_df[col] = input_df[col].astype(str).str.lower().str.replace(' ', '_')

    # 2. STANDARDIZE NUMERICAL VALUES
    input_df['years_of_experience'] = input_df['years_of_experience'].astype(float)
    input_df['age'] = input_df['age'].astype(float) 
    
    # 3. APPLY ONE-HOT ENCODING
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # 4. ALIGN COLUMNS (Create the final vector of all zeros)
    final_input = pd.DataFrame(0, index=[0], columns=features_list)
    
    # 5. MERGE DATA (Set the necessary columns to 1 or to the numerical value)
    
    # Copy all one-hot and numerical features from the input into the aligned structure
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col]

    # Explicitly set numerical values 
    if 'years_of_experience' in final_input.columns:
        final_input['years_of_experience'] = input_df['years_of_experience'].iloc[0]
        
    if 'age' in final_input.columns:
        final_input['age'] = input_df['age'].iloc[0]
            
    return final_input


# --- 2. Routes ---

@app.route('/')
def home():
    """Renders the main HTML page and provides dropdown options."""
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 'DevOps Engineer', 'Financial Analyst']
    education_levels = ['Bachelors', 'Masters', 'PhD', 'High School']
    genders = ['Male', 'Female', 'Other']
    
    return render_template('index.html', 
                            job_titles=job_titles, 
                            education_levels=education_levels, 
                            genders=genders)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the salary prediction API request."""
    try:
        data = request.get_json(force=True) 
        
        # === FIX: Clean the Input Keys ===
        # Converts all keys from the front-end (e.g., "Years of Experience")
        # to the standardized Python format (e.g., "years_of_experience")
        processed_data = {k.lower().replace(' ', '_'): v for k, v in data.items()}

        # 1. Preprocess the input data
        processed_data_df = preprocess_input(processed_data, model_features)
        
        # 2. Make prediction
        prediction = model.predict(processed_data_df)[0]
        
        # 3. Format and return response
        predicted_salary = max(0, round(prediction))
        
        response = {"predicted_salary": predicted_salary}
        
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        traceback.print_exc() 
        return jsonify({'error': 'An internal error occurred. Check input data or server logs.'}), 500

# --- 3. Run Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)