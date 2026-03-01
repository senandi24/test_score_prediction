from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessors
cat_model = joblib.load('cat_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('le_dict.pkl')

# All original features (in training order)
all_features = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance',
    'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
    'facility_rating', 'exam_difficulty'
]

# Numeric columns (for scaling)
scale_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    message = None
    color = None
    error = None

    if request.method == 'POST':
        try:
            # Get form data
            data = {}
            for f in all_features:
                if f in scale_cols:
                    data[f] = float(request.form[f])
                else:
                    data[f] = request.form[f]

            # Create DataFrame
            input_df = pd.DataFrame([data])

            # Encode categorical
            for col in le_dict:
                if col in input_df.columns:
                    input_df[col] = le_dict[col].transform(input_df[col])

            # Scale numeric
            numeric_input = input_df[scale_cols]
            scaled = scaler.transform(numeric_input)
            input_df[scale_cols] = scaled

            # Predict
            input_array = input_df[all_features].values
            pred_value = cat_model.predict(input_array)[0]
            prediction = round(float(pred_value), 2)

            # Conditional message & color
            if prediction < 40:
                message = "❌ High Risk: Student may fail."
                color = "red"
            elif prediction < 50:
                message = "⚠️ Just a Pass: Needs improvement."
                color = "orange"
            elif prediction < 70:
                message = "📘 Moderate: Keep practicing."
                color = "blue"
            else:
                message = "🌟 Excellent: Great work!"
                color = "green"

            return render_template(
                'predict.html',
                prediction=prediction,
                score=prediction,
                message=message,
                color=color
            )

        except Exception as e:
            error = str(e)
            return render_template(
                'predict.html',
                prediction=None,
                score=0,
                error=error
            )

    return render_template('predict.html', prediction=None, score=None)

if __name__ == '__main__':
    app.run(debug=True)