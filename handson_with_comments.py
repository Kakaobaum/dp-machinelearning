import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


import streamlit as st
import os

# Embed the video from the URL
video_url = "https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4"
st.video(video_url)

# Path to the presentation.md file
readme_file = "presentation.md"

# Read and render the README content
if os.path.exists(readme_file):
    with open(readme_file, "r", encoding="utf-8") as file:
        readme_content = file.read()
    st.markdown(readme_content, unsafe_allow_html=True)
else:
    st.error("presentation.md file not found!")

# Import necessary libraries:
# - Streamlit for the web application interface.
# - Pandas for handling and manipulating data.
# - NumPy for numerical computations.
# - RandomForestClassifier for the machine learning model.

# -----------------------------
# Application Description
# -----------------------------
# This Streamlit app is an interactive machine-learning-powered tool for predicting penguin species based on user-specified features. Here's how it works:
#
# 1. **Data Overview**:
#    - The app loads and displays a cleaned dataset of penguins.
#    - Users can explore the dataset, including its features and target variable.
#
# 2. **Data Visualization**:
#    - A scatter plot shows the relationship between key penguin features (`bill length` and `body mass`) grouped by species.
#
# 3. **User Input Collection**:
#    - A sidebar collects user inputs for key features such as `island`, `gender`, and measurements (`bill length`, `bill depth`, etc.).
#    - The inputs are prepared in a structured dataframe for model inference.
#
# 4. **Machine Learning Model**:
#    - A RandomForestClassifier is trained on the penguin dataset.
#    - User inputs are encoded and passed to the model for prediction.
#
# 5. **Prediction and Results**:
#    - The app outputs the predicted species along with probabilities for each possible class.
#    - Results are presented clearly using styled tables and text.
# -----------------------------

# Set the application title
st.title('We ♥️ Machine Learning')  # Displays the title of the app at the top of the page.
st.info('Deploy Python ML models with Streamlit')  # Shows an informational message about the app.

# Importing and displaying raw data
with st.expander('Data'):
    st.write('**Raw Data**')  # A section to display the raw dataset for users.
    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')  # Loads the penguins dataset from a remote URL.
    st.write(df)  # Display the dataset in a table format.

    # Display feature matrix (X)
    st.write('**X**')
    X_raw = df.drop('species', axis=1)  # Remove the target column to form the feature matrix.
    st.write(X_raw)  # Display the feature matrix.

    # Display target vector (Y)
    st.write('**Y**')
    Y_raw = df.species  # Extract the target column containing the penguin species.
    st.write(Y_raw)  # Display the target vector.

# Visualization of data
with st.expander('Visualizations'):
    # Create a scatter plot to visualize the relationship between bill length and body mass
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')  # Displays a scatter chart for visualization.

# Collecting input features from the user
with st.sidebar:  # Sidebar for user input collection.
    st.header('Input features')  # Header for the input section.
    # Input selectors for features
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))  # Dropdown for selecting island.
    gender = st.selectbox('Gender', ('male', 'female'))  # Dropdown for selecting gender.
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)  # Slider for inputting bill length.
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)  # Slider for inputting bill depth.
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)  # Slider for inputting flipper length.
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6400.0, 4207.0)  # Slider for inputting body mass.

    # Prepare input features dataframe
    data = {
        'island': island,  # User-selected island.
        'bill_length_mm': bill_length_mm,  # User-provided bill length.
        'bill_depth_mm': bill_depth_mm,  # User-provided bill depth.
        'flipper_length_mm': flipper_length_mm,  # User-provided flipper length.
        'body_mass_g': body_mass_g,  # User-provided body mass.
        'sex': gender  # User-selected gender.
    }
    input_df = pd.DataFrame(data, index=[0])  # Creates a dataframe for the user inputs.
    # Concatenate user input with raw data for consistent encoding
    input_penguins = pd.concat([input_df, X_raw], axis=0)  # Combines the user inputs with the raw dataset to ensure consistency in encoding.

# Display user input and combined data
with st.expander('Input features'):
    st.write('**Input penguin**')
    st.write(input_df)  # Display user-provided data.
    st.write('**Combined Penguin Data**')
    st.write(input_penguins)  # Display combined dataset.

# Data preparation
# Encode categorical features
encode = ['island', 'sex']  # List of categorical columns to encode.
df_penguins = pd.get_dummies(input_penguins, prefix=encode)  # One-hot encodes categorical features.
# Encoding is necessary because machine learning models cannot process categorical data directly; they require numerical representations.
input_row = df_penguins[:1]  # Extracts the encoded row for user input.
X = df_penguins[1:]  # Prepares the feature matrix by excluding the user input row.

# Encode target labels
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}  # Mapping of species names to numerical labels.
def target_encode(val):
    return target_mapper[val]  # Function to encode target values.

Y = Y_raw.apply(target_encode)  # Encodes the target vector.
# Encoding the target variable ensures that the classification model can handle the labels as numerical values, which is required for computation.

# Display encoded data
with st.expander('Data preparation'):
    st.write('**Encoded X input penguin**')
    st.write(input_row)  # Display the encoded user input.
    st.write('**Encoded Y**')
    st.write(Y)  # Display the encoded target variable.

# Model training and inference
# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier()  # Initializes the Random Forest model.
clf.fit(X, Y)  # Trains the model using the feature matrix and target vector.

# Make predictions on user input
prediction = clf.predict(input_row)  # Predicts the species for the user input.
prediction_prob = clf.predict_proba(input_row)  # Computes probabilities for each class.

# Display prediction probabilities
df_prediction_prob = pd.DataFrame(prediction_prob)  # Converts prediction probabilities to a dataframe.
df_prediction_prob.columns = ['Adelie', 'Chinstrap', 'Gentoo']  # Assigns class names to the probability columns.

# Display probabilities as a styled dataframe
st.dataframe(df_prediction_prob,
             column_config={
                 'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),  # Configures Adelie column as a progress bar.
                 'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),  # Configures Chinstrap column as a progress bar.
                 'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1)  # Configures Gentoo column as a progress bar.
             }, hide_index=True)  # Hides the index in the displayed dataframe.

# Display the predicted species
st.subheader('Predicted Species')  # Section header for displaying predictions.
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])  # Array of species names.
st.success(str(penguins_species[prediction][0]))  # Displays the predicted species as a success message.
