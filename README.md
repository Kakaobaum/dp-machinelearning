# 🎮 Machine Learning App - WS8
```
Workshop 8: Deploy Python ML models with Streamlit 🥳
```
Link for Datasets: https://github.com/dataprofessor/data
provided with ♥️ by dataprofessor.

Check out Dataprofessor on Youtube for great machine learning tutorials: 

![image](https://github.com/user-attachments/assets/57c2faf6-8063-4fb1-9058-b8274defe223) 

https://www.youtube.com/@DataProfessor 

### Class Exercise: Building a Streamlit App with the Iris Dataset

#### Objective:
In this exercise, students will deploy a machine learning model using Streamlit to classify iris species. The exercise will guide students to:
1. Load the iris dataset.
2. Collect user input for features.
3. Train and use a `RandomForestClassifier` for predictions.
---

### Step-by-Step Instructions:

1. **Import Necessary Libraries**
   - Start by importing `streamlit`, `pandas`, `numpy`, and `RandomForestClassifier` from `sklearn.ensemble`.

2. **Set Application Title an write you Name in the info**
   - Add a title and description to the Streamlit app:
     ```python
     st.title('Iris Classification 🌸')
     st.info('Classify Iris species with a Machine Learning model')
     ```

3. **Load and Display the Dataset**
   - Load the Iris dataset from the provided URL:
     ```python
     load the dataset from https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/iris.csv in a        dataframe called df
     st.write('**Iris Dataset**')
     st.write(df)
     ```

4. **Set Up User Input (Sidebar)**
   - create a sidebar and Collect feature values from the user using sliders:
         sepal_length min 4.3, max 7.9
         sepal_width min 2.0, max 4.4
         petal_length min 1.0, max 6.9
         petal_width min 0.1, max 2.5
     !! don't forget to add default values between min and max
add the following
```python
         input_data = {
             'sepal_length': sepal_length,
             'sepal_width': sepal_width,
             'petal_length': petal_length,
             'petal_width': petal_width
         }
         input_df = pd.DataFrame(input_data, index=[0])
     ```

5. **Data Preparation**
   - Separate features (X) and target labels (Y):
     ```python
     X = df.drop('Species', axis=1)
     Y = df['Species']
     ```

   - Encode target labels:
     ```python
     target_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
     Y_encoded = Y.map(target_mapping)
     ```

6. **Train the RandomForest Model**
   - Initialize and train the model:
     ```python
     clf = RandomForestClassifier()
     clf.fit(X, Y_encoded)
     ```

7. **Make Predictions**
   - Predict species based on user input:
     ```python
     prediction = clf.predict(input_df)
     prediction_proba = clf.predict_proba(input_df)
     ```

8. **Display Results**
   - Show prediction probabilities and the predicted species:
     ```python
     st.write('**Prediction Probabilities**')
     prob_df = pd.DataFrame(prediction_proba, columns=['Setosa', 'Versicolor', 'Virginica'])
     st.write(prob_df)
```
# Display the predicted species
     add subheader - Predicted species
     ```python
     species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
     ```
     output of the prediction
     

---

### Deliverables for Students:
- A fully functional Streamlit app that can classify iris species based on user input.

---

### Exercise Goal:
Students learn how to load data, create a machine learning pipeline, collect user inputs, and display predictions using Streamlit in under 10 minutes!
