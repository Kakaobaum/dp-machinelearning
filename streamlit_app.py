import streamlit as st
import pandas as pd

st.title('We ♥️ Machine Learning')

st.info('Deploy Python ML models with Streamlit')

with st.expaner('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
