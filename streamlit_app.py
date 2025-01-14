import streamlit as st
import pandas as pd

st.title('We ♥️ Machine Learning')

st.info('Deploy Python ML models with Streamlit')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**Y**')
  Y = df.species
  Y

#"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
with st.expander('Visualizations')
  st.scatter_chart(data=df,x = 'bill_length_mm', y = 'body_mass_g', color = species)
