# Live Coding: Deploy Python ML Models with Streamlit
# Goal: Deploy a RandomForestClassifier to predict penguin species based on user inputs

# Step 1: Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 2: Set up the Streamlit application
st.title('We ❤️ Machine Learning')
st.info('Deploy Python ML models with Streamlit')

# Step 3: Load and display the dataset
with st.expander('Data'):
    st.write('**Raw Data**')
    # Load penguins dataset from a URL
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    st.write(df)  # Display the dataset

    # Separate features (X) and target variable (Y)
    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.write(X_raw)

    st.write('**Y**')
    Y_raw = df.species
    st.write(Y_raw)

# Step 4: Add a visualization
with st.expander('Visualizations'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Step 5: Create a sidebar for user inputs
with st.sidebar:
    st.header('Input features')
    # Add inputs for all necessary features
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    gender = st.selectbox('Gender', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6400.0, 4207.0)

    # Combine inputs into a DataFrame
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])
    # Append to original data for consistent encoding
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Step 6: Show user inputs
with st.expander('Input features'):
    st.write('**Input penguin**')
    st.write(input_df)
    st.write('**Combined Penguin Data**')
    st.write(input_penguins)

# Step 7: Prepare the data
encode = ['island', 'sex']  # Categorical columns to encode
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row = df_penguins[:1]  # Extract encoded user input row
X = df_penguins[1:]  # Encoded feature matrix

# Encode the target variable
st.write('Encoding target labels...')
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

Y = Y_raw.apply(target_encode)

# Step 8: Train the Random Forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Step 9: Make predictions
prediction = clf.predict(input_row)  # Predicted label
prediction_prob = clf.predict_proba(input_row)  # Prediction probabilities

# Step 10: Display predictions
# Show prediction probabilities
st.info('Prediction probabilities:')
df_prediction_prob = pd.DataFrame(prediction_prob, columns=['Adelie', 'Chinstrap', 'Gentoo'])
st.dataframe(df_prediction_prob,
             column_config={
                 'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
                 'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
                 'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1)
             }, hide_index=True)

# Show the predicted species
st.subheader('Predicted Species')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
