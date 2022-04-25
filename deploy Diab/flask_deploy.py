import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
model = joblib.load('RF.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    if request.method == "POST":
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])
        data = np.array([[pregnancies,glucose,blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function,age]])
        prediction = model.predict(data)
        print(prediction)
        output = prediction[0]
        print(output)
        if output == 0:
            output ='You Are Not Affected With Diabetes'
        elif output == 1:
            output ='You Are Affected With Diabetes'            
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(host="localhost", port=8000)