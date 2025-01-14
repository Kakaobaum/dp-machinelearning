import streamlit as st
import pandas as pd

st.title('We ♥️ Machine Learning')

st.info('Deploy Python ML models with Streamlit')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df
