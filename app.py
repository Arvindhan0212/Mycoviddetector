from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
app = Flask(__name__)

def predict_covid_status(user_input_array, model='ensemble'):
    data_file = 'static/data/Covid_Detector.csv'
    data = pd.read_csv(data_file)
    
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':  # Only encode object types
            data[column] = le.fit_transform(data[column])
    
    X = data.drop(columns='COVID-19')
    y = data['COVID-19']
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    if model == 'ensemble':
        with open('ensemble_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        predictions = loaded_model.predict(pd.DataFrame(user_input_array, columns=X.columns))
        results = ["You may be at risk for COVID-19, please contact a medical professional." if pred == 1 else "You do not display any symptoms of COVID-19." for pred in predictions]
        positive = [pred == 1 for pred in predictions]
    
    return results, positive

@app.route('/')
def index():
    file_path = "static/data/Covid_Detector.csv"
    data = pd.read_csv(file_path)
    columns = data.drop(columns='COVID-19').columns.tolist()
    return render_template('index.html', columns=columns)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form.to_dict()
        model_choice = user_input.pop('model')
        user_input = {key: int(value) for key, value in user_input.items()}
        user_input_array = [user_input]
        results, positive = predict_covid_status(user_input_array, model=model_choice)
        return render_template('result.html', prediction=results[0], positive=positive[0])

if __name__ == '__main__':
    app.run(debug=True)
