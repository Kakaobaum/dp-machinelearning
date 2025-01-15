# TODO: import streamlit, pandas, numpy, and RandomForestClassifier from sklearn.ensemble.
# add code here...

# TODO: Set personalized application title an write your name in the info
# add code here...

# TODO: Load and display dataset from https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/iris.csv as a dataframe
# add your code here...

# Sidebar inputs
# TODO: create a sidebar and Collect feature values from the user using sliders:
#sepal_length min 4.3, max 7.9
#sepal_width min 2.0, max 4.4
#petal_length min 1.0, max 6.9
#petal_width min 0.1, max 2.5
#! don't forget to add default values between min and max 
# add your code here...

''' now you can delete this line
    input_data = {
        'Sepal.Length': sepal_length,
        'Sepal.Width': sepal_width,
        'Petal.Length': petal_length,
        'Petal.Width': petal_width
    }
    input_df = pd.DataFrame(input_data, index=[0])

# Data preparation

X = df.drop('Species', axis=1)  # Correct column name
Y = df['Species']  # Correct column name
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
''' '''now you can delete this line '''


#! don't forget to delete the triple-quoted strings


# TODO: Display the predicted species
# species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
# add code here...


