from flask import Flask, request, render_template, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    result = species[prediction[0]]
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
