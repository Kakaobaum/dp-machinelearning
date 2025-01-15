# Live Coding: Deploy Python ML Models with Streamlit
# Goal: Deploy a RandomForestClassifier to predict penguin species based on user inputs

# Step 1: Import required libraries
# TODO: Import required libraries
# add code here...

# Step 2: Set up the Streamlit application
# Add title and description for your application
# add code here...
# add code here...

# Step 3: Load and display the dataset
# TODO: add expander with Title
# add code here...
    # TODO: Load dataset here using pd.read_csv and display it
    # add code here...

    # Display feature matrix (X)
    #st.write('**X**')
    #X_raw = df.drop('species', axis=1)  # Remove the target column to form features
    #st.write(X_raw)

    # Display target vector (Y)
    #st.write('**Y**')
    #Y_raw = df.species  # Define the target variable
    #st.write(Y_raw)

# Step 4: Add a visualization
with st.expander('Visualizations'):
    # TODO: Add a scatter chart to visualize the data
    # # add code here...

# Step 5: Create a sidebar for user inputs
with st.sidebar:
    st.header('Input features')
    # TODO: Add feature inputs (selectbox for island, selectbox for gender, sliders for bill length(min. 32.1,max 59.6, default value))
    # island = add code here...
    # gender = add code here...
    # bill_length_mm = add code here...
    # bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    # flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    # body_mass_g = st.slider('Body Mass (g)', 2700.0, 6400.0, 4207.0)

    # TODO: Combine inputs into a DataFrame
    # data = {
    #     'island': island,
    #     'bill_length_mm': bill_length_mm,
    #     'bill_depth_mm': bill_depth_mm,
    #     'flipper_length_mm': flipper_length_mm,
    #     'body_mass_g': body_mass_g,
    #     'sex': gender
    # }
    # input_df = pd.DataFrame(data, index=[0])
    # input_penguins = pd.concat([input_df, X_raw], axis=0)

# Step 6: Show user inputs
with st.expander('Input features'):
    # TODO: Display user inputs and combined data
    # st.write('**Input penguin**')
    # st.write(input_df)
    # st.write('**Combined Penguin Data**')
    # st.write(input_penguins)

# Step 7: Prepare the data
# TODO: Encode categorical variables and prepare X and Y
# encode = ['island', 'sex']
# df_penguins = pd.get_dummies(input_penguins, prefix=encode)
# input_row = df_penguins[:1]
# X = df_penguins[1:]

# TODO: Encode the target variable
# target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
# def target_encode(val):
#     return target_mapper[val]
# Y = Y_raw.apply(target_encode)

# Step 8: Train the Random Forest model
# TODO: Initialize and fit the RandomForestClassifier
# add code here...
# add code here...

# Step 9: Make predictions
# TODO: Add prediction and probability code
# prediction = clf.predict(input_row)
# prediction_prob = clf.predict_proba(input_row)

# Step 10: Display predictions
# TODO: Show prediction probabilities and species using a ProgressColumn
# df_prediction_prob = pd.DataFrame(prediction_prob, columns=['Adelie', 'Chinstrap', 'Gentoo'])
# add code here...

    
# Step 11: Display just the predicted Species.
# st.subheader('Predicted Species')
# add code here...
# add code here...
