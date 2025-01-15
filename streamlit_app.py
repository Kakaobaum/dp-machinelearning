import streamlit as st
import pandas as pd

st.title('We ♥️ Machine Learning')

st.info('Deploy Python ML models with Streamlit')

#importing Data
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  # model parameters(training data)
  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw
  # We will predict the species
  st.write('**Y**')
  Y_raw = df.species
  Y_raw

#"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
with st.expander('Visualizations'):
  st.scatter_chart(data=df,x = 'bill_length_mm', y = 'body_mass_g', color = 'species')

# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  gender = st.selectbox('Gender',('male','female'))
  bill_length_mm = st.slider('Bill length (mm)',32.1,59.6,43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1,21.5,17.2)
  flipper_length_mm = st.slider('Flipper length (mm)',172.0,231.0,201.0)
  body_mass_g = st.slider('Body Mass (g)',2700.0,6400.0,4207.0)

  # create df for the input features
  data={'island': island,
        'bill_length_mm': bill_length_mm, 
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender}
  #input parameters
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis = 0)
  

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Conbined Penguin Data**')
  input_penguins  

#Data preparation
#Encode X
encode = ['island', 'sex']
df_penguins=pd.get_dummies(input_penguins,prefix=encode)
input_row = df_penguins[:1]

#Encode Y
target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
  return target_mapper[val]

Y = Y_raw.apply(target_encode)


with st.expander('Data preparation'):
  st.write('**Encoded input penguin**')
  input_row  
  st.write('**Encoded Y**')
  Y
