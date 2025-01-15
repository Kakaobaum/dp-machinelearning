import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Iris Classification 🌸')
st.info('Classify Iris species with a Machine Learning model')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/iris.csv')
st.write('**Iris Dataset**')
st.write(df)

# Sidebar inputs
with st.sidebar:
    st.header('Input features')
    sepal_length = st.slider('Sepal Length (cm)', 4.3, 7.9, 5.8)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.4, 3.0)
    petal_length = st.slider('Petal Length (cm)', 1.0, 6.9, 4.35)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

    input_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    input_df = pd.DataFrame(input_data, index=[0])

# Data preparation
X = df.drop('species', axis=1)
Y = df['species']
target_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
Y_encoded = Y.map(target_mapping)

# Train model
clf = RandomForestClassifier()
clf.fit(X, Y_encoded)

# Predict
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display results
st.write('**Prediction Probabilities**')
prob_df = pd.DataFrame(prediction_proba, columns=['Setosa', 'Versicolor', 'Virginica'])
st.write(prob_df)

st.subheader('Predicted Species')
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
st.success(species_mapping[prediction[0]])
