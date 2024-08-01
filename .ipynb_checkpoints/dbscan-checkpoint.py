import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/arvindhan/Downloads/archive-2/Covid_Detector.csv'
data = pd.read_csv(file_path)

# Initialize label encoder
le = LabelEncoder()

# Convert categorical data to numerical data
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Separate features and target
X = data.drop(columns='COVID-19')
y = data['COVID-19']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data based on COVID-19 status
X_covid_positive = X_scaled[y == 1]
X_covid_negative = X_scaled[y == 0]

# Apply DBSCAN
dbscan_positive = DBSCAN(eps=0.5, min_samples=5)
clusters_positive = dbscan_positive.fit_predict(X_covid_positive)

dbscan_negative = DBSCAN(eps=0.5, min_samples=5)
clusters_negative = dbscan_negative.fit_predict(X_covid_negative)

# Example user input (change as needed)
user_input = {
    'Breathing Problem': 1,
    'Fever': 1,
    'Dry Cough': 1,
    'Sore throat': 1,
    'Running Nose': 1,
    'Asthma': 0,  # Optional, depending on comorbidity status
    'Chronic Lung Disease': 0,  # Optional, depending on comorbidity status
    'Headache': 1,
    'Heart Disease': 0,  # Optional, depending on comorbidity status
    'Diabetes': 0,  # Optional, depending on comorbidity status
    'Hyper Tension': 0,  # Optional, depending on comorbidity status
    'Fatigue ': 1,
    'Gastrointestinal ': 1,
    'Abroad travel': 1,
    'Contact with COVID Patient': 1,
    'Attended Large Gathering': 1,
    'Visited Public Exposed Places': 1,
    'Family working in Public Exposed Places': 1,
    'Wearing Masks': 0,  # Indicates not wearing masks, higher risk
    'Sanitization from Market': 0  # Indicates not sanitizing, higher risk
}


# Convert user input to DataFrame and scale
input_df = pd.DataFrame([user_input], columns=X.columns)
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

# Function to find the closest cluster to the user input
def find_closest_cluster(input_scaled, X_scaled, clusters):
    from sklearn.metrics import pairwise_distances_argmin_min
    closest_cluster_idx, _ = pairwise_distances_argmin_min(input_scaled, X_scaled)
    return clusters[closest_cluster_idx[0]]

# Function to get the majority class in a cluster
def get_cluster_label(clusters, y):
    cluster_labels = {}
    for cluster in np.unique(clusters):
        if cluster != -1:  # Ignore noise
            cluster_labels[cluster] = y[clusters == cluster].mode()[0]
    return cluster_labels

# Get cluster labels
positive_cluster_labels = get_cluster_label(clusters_positive, y[y == 1])
negative_cluster_labels = get_cluster_label(clusters_negative, y[y == 0])

# Predict the class based on the majority class in the cluster
def predict_class(cluster, cluster_labels):
    return cluster_labels.get(cluster, -1)  # Return -1 if cluster not found

# Generate predictions for the entire dataset
def generate_predictions(X_scaled, clusters, cluster_labels):
    cluster_predictions = np.array([predict_class(cluster, cluster_labels) for cluster in clusters])
    return cluster_predictions

# Get predictions for positive and negative cases
positive_predictions = generate_predictions(X_covid_positive, clusters_positive, positive_cluster_labels)
negative_predictions = generate_predictions(X_covid_negative, clusters_negative, negative_cluster_labels)

# Combine predictions and true labels
combined_predictions = np.concatenate([positive_predictions, negative_predictions])
combined_true_labels = np.concatenate([np.ones_like(positive_predictions), np.zeros_like(negative_predictions)])

# Compute confusion matrix
conf_matrix = confusion_matrix(combined_true_labels, combined_predictions, labels=[0, 1])

# Display the confusion matrix
def plot_confusion_matrix(conf_matrix, title):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, 'Confusion Matrix for COVID-19 Clusters')

# Display the result
user_prediction = predict_class(find_closest_cluster(input_scaled, X_scaled, np.concatenate([clusters_positive, clusters_negative])), 
                                 {**positive_cluster_labels, **negative_cluster_labels})

if user_prediction == 1:
    print("User is predicted to be COVID-19 positive.")
else:
    print("User is predicted to be COVID-19 negative.")
with open('ensemble.pkl', 'rb') as file:
    loaded_ensemble = pickle.load(file)