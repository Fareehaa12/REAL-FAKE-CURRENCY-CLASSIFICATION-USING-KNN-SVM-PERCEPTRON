# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:19:49 2024

@author: HP
"""

# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved models using pickle
knn_model = pickle.load(open('C:/AI&ES LABS/LAB 9 ATTACHMENTS/realfakecurrency_knn_model.sav', 'rb'))
svm_model = pickle.load(open('C:/AI&ES LABS/LAB 9 ATTACHMENTS/realfakecurrency_svm_model.sav', 'rb'))
perceptron_model = pickle.load(open('C:/AI&ES LABS/LAB 9 ATTACHMENTS/realfakecurrency_perceptron_model.sav', 'rb'))

# Function to standardize input data
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Streamlit interface
st.title('Real/Fake Currency Note Classification')

st.write("This application allows you to classify a currency note as real or fake based on the input features.")

# Input features
variance = st.number_input('Variance', min_value=-10.0, max_value=10.0, value=0.0)
skewness = st.number_input('Skewness', min_value=-10.0, max_value=10.0, value=0.0)
curtosis = st.number_input('Curtosis', min_value=-10.0, max_value=10.0, value=0.0)
entropy = st.number_input('Entropy', min_value=-10.0, max_value=10.0, value=0.0)

# Convert the input to a dataframe
input_data = np.array([[variance, skewness, curtosis, entropy]])
input_df = pd.DataFrame(input_data, columns=['variance', 'skewness', 'curtosis', 'entropy'])

# Standardize the input data
input_scaled = standardize_data(input_df)

# Prediction
if st.button('Classify'):
    knn_pred = knn_model.predict(input_scaled)
    svm_pred = svm_model.predict(input_scaled)
    perceptron_pred = perceptron_model.predict(input_scaled)

    # Show the predictions
    st.write("KNN Prediction: ", "Real" if knn_pred[0] == 1 else "Fake")
    st.write("SVM Prediction: ", "Real" if svm_pred[0] == 1 else "Fake")
    st.write("Perceptron Prediction: ", "Real" if perceptron_pred[0] == 1 else "Fake")
    
# Additional visualization for confusion matrix can be added if required.
