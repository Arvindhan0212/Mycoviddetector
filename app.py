from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min

app = Flask(__name__)

def predict_covid_status(user_input_array, model='ensemble'):
    data_file = 'static/data/Covid_Detector.csv'
    
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return ["An error occurred during prediction."], [False]

    # Encode categorical features
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':  # Only encode object types
            data[column] = le.fit_transform(data[column])
    
    X = data.drop(columns='COVID-19')
    y = data['COVID-19']
    
    # Preprocess data
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    results = []
    positive = []
    
    try:
        if model == 'ensemble':
            # Load and use ensemble model
            with open('ensemble_model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            predictions = loaded_model.predict(pd.DataFrame(user_input_array, columns=X.columns))
            results = ["You may be at risk for COVID-19, please contact a medical professional." if pred == 1 else "You do not display any symptoms of COVID-19." for pred in predictions]
            positive = [pred == 1 for pred in predictions]

        elif model == 'dbscan':
            # Load and use DBSCAN model
            with open('dbscan2.pkl', 'rb') as file:
                data = pickle.load(file)
                dbscan = data['dbscan']
                scaler = data['scaler']
                imputer = data['imputer']
                cluster_labels = data['cluster_labels']
            
            # Process user input
            input_df = pd.DataFrame(user_input_array, columns=X.columns)
            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)

            def find_closest_cluster(input_scaled, X_scaled, clusters):
                closest_cluster_idx, _ = pairwise_distances_argmin_min(input_scaled, X_scaled)
                return clusters[closest_cluster_idx[0]]

            # Predict clusters for user input
            user_clusters = []
            for input_row in input_scaled:
                user_cluster = find_closest_cluster([input_row], X_scaled, np.unique(dbscan.labels_))
                user_clusters.append(user_cluster)
            
            for user_cluster in user_clusters:
                user_prediction = cluster_labels.get(user_cluster, -1) if user_cluster != -1 else -1
                if user_prediction == 1:
                    results.append("You may be at risk for COVID-19, please contact a medical professional.")
                    positive.append(True)
                else:
                    results.append("You do not display any symptoms of COVID-19.")
                    positive.append(False)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        results.append("An error occurred during prediction.")
        positive.append(False)

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
        try:
            results, positive = predict_covid_status(user_input_array, model=model_choice)
            return render_template('result.html', prediction=results[0], positive=positive[0])
        except Exception as e:
            print(f"Error in /predict route: {e}")
            return render_template('result.html', prediction="An error occurred during prediction.", positive=False)

if __name__ == '__main__':
    app.run(debug=True)
