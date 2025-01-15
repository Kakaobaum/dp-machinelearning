# 🎮 Machine Learning App - WS8
```
Workshop 8: Deploy Python ML models with Streamlit 🥳
```
Link for Datasets: https://github.com/dataprofessor/data by dataprofessor 🤓.

Check out Dataprofessor on Youtube for great machine learning tutorials: 

![image](https://github.com/user-attachments/assets/57c2faf6-8063-4fb1-9058-b8274defe223) 

https://www.youtube.com/@DataProfessor 

### Class Exercise: Building a Streamlit App with the Iris Dataset

#### Objective:
In this exercise, students will deploy a machine learning model using Streamlit to classify iris species. The exercise will guide students to:
1. Load the iris dataset.
2. Collect user input for features.
3. Train and use a `RandomForestClassifier` for predictions.

✍️ complete the TODOs in class_exercise.py 
please delete the triple-quoted strings when you reach the code lines that instruct to do so 

--

### Step-by-Step Instructions:

1. **Import Necessary Libraries**
   - Start by importing `streamlit`, `pandas`, `numpy`, and `RandomForestClassifier` from `sklearn.ensemble`.

2. **Set Application Title**
   - Add a title and description to the Streamlit app:

3. **Load and Display the Dataset**
   - Load the Iris dataset from the provided URL: https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/iris.csv

4. **Set Up User Input (Sidebar)**
   - Collect feature values from the user using sliders and dropdowns:

5. **Data Preparation**
   - Separate features (X) and target labels (Y):
     ```python
     X = df.drop('species', axis=1)
     Y = df['species']
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
