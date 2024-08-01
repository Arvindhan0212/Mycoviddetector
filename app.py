from flask import Flask, request, render_template, url_for
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min

app = Flask(__name__)

def predict_covid_status(user_input_array, model='ensemble'):
    data_file = 'static/Covid_Detector.csv'
    data = pd.read_csv(data_file)
    le = LabelEncoder()
    for column in data.columns:
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
    elif model == 'dbscan':
        with open('dbscan.pkl', 'rb') as file:
            loaded_ensemble = pickle.load(file)

        dbscan_positive = loaded_ensemble['dbscan_positive']
        dbscan_negative = loaded_ensemble['dbscan_negative']
        clusters_positive = loaded_ensemble['clusters_positive']
        clusters_negative = loaded_ensemble['clusters_negative']
        positive_cluster_labels = loaded_ensemble['positive_cluster_labels']
        negative_cluster_labels = loaded_ensemble['negative_cluster_labels']

        input_df = pd.DataFrame(user_input_array, columns=X.columns)
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        def find_closest_cluster(input_scaled, X_scaled, clusters):
            closest_cluster_idx, _ = pairwise_distances_argmin_min(input_scaled, X_scaled)
            return clusters[closest_cluster_idx[0]]

        def get_cluster_label(clusters, y):
            cluster_labels = {}
            for cluster in np.unique(clusters):
                if cluster != -1:
                    cluster_labels[cluster] = y[clusters == cluster].mode()[0]
            return cluster_labels

        def predict_class(cluster, cluster_labels):
            return cluster_labels.get(cluster, -1)

        results = []
        positive = []
        for input_row in input_scaled:
            user_cluster = find_closest_cluster([input_row], X_scaled, np.concatenate([clusters_positive, clusters_negative]))
            user_prediction = predict_class(user_cluster, {**positive_cluster_labels, **negative_cluster_labels})
            if user_prediction != 1:
                results.append("You may be at risk for COVID-19, please contact a medical professional.")
                positive.append(True)
            else:
                results.append("You do not display any symptoms of COVID-19.")
                positive.append(False)

    return results, positive

@app.route('/')
def index():
    file_path = "static/Covid_Detector.csv"
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
