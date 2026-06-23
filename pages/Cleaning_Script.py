import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.title("Data Cleaning Dashboard")

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return None

def parse_years_to_omit(year_input):
    if not year_input.strip():
        return []

    years = set()

    for part in year_input.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start_year = int(start_str.strip())
            end_year = int(end_str.strip())

            if start_year > end_year:
                raise ValueError(f"Invalid range: {part}. Start year must be less than or equal to end year.")

            years.update(range(start_year, end_year + 1))
        else:
            years.add(int(part))

    return sorted(years)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def clear_cleaning_state():
    keys_to_clear = [
        "uploaded_file",
        "year_col",
        "value_col",
        "years_omit_input",
        "cleaned_df"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.subheader("File Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    key="uploaded_file"
)

if uploaded_file is not None:
    df = load_file(uploaded_file)

    if df is not None:
        st.subheader("Raw Data Preview")
        st.write("First 5 rows:")
        st.dataframe(df.head())

        st.write("Last 5 rows:")
        st.dataframe(df.tail())

        st.write(f"Total rows: {len(df)}")

        st.subheader("Data Cleaning")

        columns = df.columns.tolist()

        year_col = st.selectbox(
            "Select the year/date column",
            options=columns,
            key="year_col"
        )

        value_options = [col for col in columns if col != year_col]
        value_col = st.selectbox(
            "Select the value column",
            options=value_options,
            key="value_col"
        )

        years_omit_input = st.text_input(
            "Type years or year ranges to omit (e.g. 2016 or 2010-2016 or 2010-2016,2019)",
            key="years_omit_input"
        )

        if st.button("Clean Data"):
            try:
                years_to_omit = parse_years_to_omit(years_omit_input)

                filter_years = pd.to_datetime(df[year_col], errors="coerce").dt.year

                cleaned_df = df.loc[
                    ~filter_years.isin(years_to_omit),
                    [year_col, value_col]
                ].copy()

                cleaned_df = cleaned_df.rename(columns={
                    year_col: "Date",
                    value_col: "Value"
                })

                cleaned_df = cleaned_df[["Date", "Value"]]

                cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"], errors="coerce")
                cleaned_df = cleaned_df.sort_values(by="Date", ascending=True).reset_index(drop=True)

                cleaned_df = cleaned_df[["Date", "Value"]]

                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["years_to_omit"] = years_to_omit

            except Exception as e:
                st.error(f"Error: {e}")

        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]

            st.subheader("Cleaned Data Preview")
            st.write("First 5 rows:")
            st.dataframe(cleaned_df.head())

            st.write("Last 5 rows:")
            st.dataframe(cleaned_df.tail())

            if "years_to_omit" in st.session_state:
                st.write(f"Years omitted: {st.session_state['years_to_omit']}")

            st.write(f"Rows retained: {len(cleaned_df)}")

            original_name = uploaded_file.name
            base_name = os.path.splitext(original_name)[0]
            output_file_name = f"{base_name}_cleaned.csv"

            st.download_button(
                label="Download Cleaned CSV",
                data=convert_df_to_csv(cleaned_df),
                file_name=output_file_name,
                mime="text/csv"
            )
    else:
        st.error("Unsupported file type.")

st.write("---")

if st.button("Clear All"):
    clear_cleaning_state()