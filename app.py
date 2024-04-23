from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('credit_card_predictions.db')
    print("Opened database successfully")

    conn.execute('''CREATE TABLE IF NOT EXISTS predictions
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             code_gender TEXT,
             flag_own_car TEXT,
             flag_own_reality TEXT,
             cnt_children INTEGER,
             amt_income_total REAL,
             name_income_type TEXT,
             name_education_type TEXT,
             name_family_status TEXT,
             name_housing_type TEXT,
             days_birth DATE,
             days_employed INTEGER,
             flag_mobil INTEGER,
             flag_work_phone INTEGER,
             flag_phone INTEGER,
             flag_email INTEGER,
             occupation_type TEXT,
             cnt_fam_members INTEGER,
             age INTEGER,
             prediction INTEGER,
             probability REAL);''')
    print("Table created successfully")
    conn.close()

init_db()
# Load the pre-trained model
model = pickle.load(open('credit_card_approval_model.pkl', 'rb'))

# Define mappings
income_type_mapping = {
    'Working': 0,
    'Commercial associate': 1,
    'Pensioner': 2,
    'State servant': 3,
    'Student': 4
}

education_mapping = {
    'Higher education': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Lower secondary': 3,
    'Academic degree': 4
}

family_status_mapping = {
    'Civil marriage': 0,
    'Married': 1,
    'Single / not married': 2,
    'Separated': 3,
    'Widow': 4
}

housing_type_mapping = {
    'Rented apartment': 0,
    'House / apartment': 1,
    'Municipal apartment': 2,
    'With parents': 3,
    'Co-op apartment': 4,
    'Office apartment': 5
}

occupation_mapping = {
    'Security staff': 0,
    'Sales staff': 1,
    'Accountants': 2,
    'Laborers': 3,
    'Managers': 4,
    'Drivers': 5,
    'Core staff': 6,
    'High skill tech staff': 7,
    'Cleaning staff': 8,
    'Private service staff': 9,
    'Cooking staff': 10,
    'Low-skill Laborers': 11,
    'Waiters/barmen staff': 12,
    'Medicine staff': 13,
    'Secretaries': 14,
    'HR staff': 15,
    'IT staff': 16
}

@app.route('/')
def home():
    return render_template('index.html')
from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    data = {
        'CODE_GENDER': request.form['CODE_GENDER'],
        'FLAG_OWN_CAR': request.form['FLAG_OWN_CAR'],
        'FLAG_OWN_REALITY': request.form['FLAG_OWN_REALITY'],
        'CNT_CHILDREN': int(request.form['CNT_CHILDREN']),
        'AMT_INCOME_TOTAL': float(request.form['AMT_INCOME_TOTAL']),
        'NAME_INCOME_TYPE': request.form['NAME_INCOME_TYPE'],
        'NAME_EDUCATION_TYPE': request.form['NAME_EDUCATION_TYPE'],
        'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS'],
        'NAME_HOUSING_TYPE': request.form['NAME_HOUSING_TYPE'],
        'DAYS_BIRTH': datetime.strptime(request.form['DAYS_BIRTH'], '%Y-%m-%d').date(),
        'DAYS_EMPLOYED': int(request.form['DAYS_EMPLOYED']),
        'FLAG_MOBIL': int(request.form['FLAG_MOBIL']),
        'FLAG_WORK_PHONE': int(request.form['FLAG_WORK_PHONE']),
        'FLAG_PHONE': int(request.form['FLAG_PHONE']),
        'FLAG_EMAIL': int(request.form['FLAG_EMAIL']),
        'OCCUPATION_TYPE': request.form['OCCUPATION_TYPE'],
        'CNT_FAM_MEMBERS': int(request.form['CNT_FAM_MEMBERS'])
    }

    # Calculate age from the date of birth
    today = datetime.now().date()
    age = today.year - data['DAYS_BIRTH'].year - ((today.month, today.day) < (data['DAYS_BIRTH'].month, data['DAYS_BIRTH'].day))

    # Add age to data dictionary
    data['AGE'] = age

    # Convert categorical data to numerical format using mappings
    data['CODE_GENDER'] = 1 if data['CODE_GENDER'] == 'M' else 0
    data['FLAG_OWN_CAR'] = 1 if data['FLAG_OWN_CAR'] == 'Yes' else 0
    data['FLAG_OWN_REALITY'] = 1 if data['FLAG_OWN_REALITY'] == 'Yes' else 0
    data['NAME_INCOME_TYPE'] = income_type_mapping[data['NAME_INCOME_TYPE']]
    data['NAME_EDUCATION_TYPE'] = education_mapping[data['NAME_EDUCATION_TYPE']]
    data['NAME_FAMILY_STATUS'] = family_status_mapping[data['NAME_FAMILY_STATUS']]
    data['NAME_HOUSING_TYPE'] = housing_type_mapping[data['NAME_HOUSING_TYPE']]
    data['OCCUPATION_TYPE'] = occupation_mapping[data['OCCUPATION_TYPE']]

    # Prepare input features for prediction
    features = np.array([
        data['CODE_GENDER'], data['FLAG_OWN_CAR'], data['FLAG_OWN_REALITY'],
        data['CNT_CHILDREN'], data['AMT_INCOME_TOTAL'], data['NAME_INCOME_TYPE'],
        data['NAME_EDUCATION_TYPE'], data['NAME_FAMILY_STATUS'], data['NAME_HOUSING_TYPE'],
        data['AGE'], data['DAYS_EMPLOYED'], data['FLAG_MOBIL'], data['FLAG_WORK_PHONE'],
        data['FLAG_PHONE'], data['FLAG_EMAIL'], data['OCCUPATION_TYPE'], data['CNT_FAM_MEMBERS'],
        0, 0, 0, 0, 0, 0  # Add placeholders for the missing features
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    prediction_probability = model.predict_proba(features)

    conn = sqlite3.connect('credit_card_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''INSERT INTO predictions
                    (code_gender, flag_own_car, flag_own_reality, cnt_children,
                    amt_income_total, name_income_type, name_education_type,
                    name_family_status, name_housing_type, days_birth, days_employed,
                    flag_mobil, flag_work_phone, flag_phone, flag_email, occupation_type,
                    cnt_fam_members, age, prediction, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (data['CODE_GENDER'], data['FLAG_OWN_CAR'], data['FLAG_OWN_REALITY'], data['CNT_CHILDREN'],
                    data['AMT_INCOME_TOTAL'], data['NAME_INCOME_TYPE'], data['NAME_EDUCATION_TYPE'],
                    data['NAME_FAMILY_STATUS'], data['NAME_HOUSING_TYPE'], data['DAYS_BIRTH'],
                    data['DAYS_EMPLOYED'], data['FLAG_MOBIL'], data['FLAG_WORK_PHONE'], data['FLAG_PHONE'],
                    data['FLAG_EMAIL'], data['OCCUPATION_TYPE'], data['CNT_FAM_MEMBERS'], data['AGE'],
                    int(prediction[0]), float(prediction_probability[0][0])))

    conn.commit()
    conn.close()


    # Return prediction as a response
    return render_template('result.html', prediction=prediction[0], probability=prediction_probability[0])

if __name__ == '__main__':
    app.run(debug=True)
