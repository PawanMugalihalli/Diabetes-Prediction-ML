from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('logreg.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Obtain form data
    Pregnancies = float(request.form.get('pregnancy'))
    Glucose = float(request.form.get('glucose'))
    BloodPressure = float(request.form.get('bp'))
    SkinThickness = float(request.form.get('skinthickness'))
    Insulin = float(request.form.get('insulin'))
    BMI = float(request.form.get('bmi'))
    DiabetesPedigreeFunction = float(request.form.get('pedigreefunction'))
    Age = float(request.form.get('age'))

    # Make prediction
    prediction = model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))
    
    # Determine prediction result
    result = "Positive" if prediction[0] == 1 else "Negative"
    
    # Send JSON response
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

