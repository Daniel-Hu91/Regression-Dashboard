import streamlit as st

st.set_page_config(page_title="Home", layout="wide")
st.title("Regression Analysis Dashboard")
st.write("Navigate to the various pages on the left")
st.write("The Cleaning Script takes a data file and sorts it from earliest date to latest date. It also has the option to filter by years only. Additionally, it cleans the file to only include the date and value columns.")
st.write("The Regression Analysis has the option to either run a time trend regression or linear regression. Furthermore, for the linear regression there is the option to run the regression for all time periods or for each specific year. There is the option to remove outliers for both regressions.")