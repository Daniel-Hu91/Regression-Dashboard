import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regression Analysis", layout="wide")
st.title("Regression Analysis")

if "regression_uploader_key" not in st.session_state:
    st.session_state["regression_uploader_key"] = 0

def clear_regression_state():
    keys_to_clear = [
        "time_trend_results",
        "linear_regression_results"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["regression_uploader_key"] += 1
    st.rerun()

def remove_outliers_iqr(df, columns):
    filtered_df = df.copy()

    for col in columns:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_df = filtered_df[
            (filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)
        ]

    return filtered_df

st.write("Use the tabs below to choose a regression type.")

tab1, tab2 = st.tabs(["Time Trend Regression", "Linear Regression"])

# ---------------------------------------------------
# TIME TREND REGRESSION TAB
# ---------------------------------------------------
with tab1:
    st.subheader("Time Trend Regression")
    st.write(
        "Upload one cleaned file with columns `date` and `value` to estimate "
        "whether the variable trends upward or downward over time."
    )

    time_file = st.file_uploader(
        "Upload cleaned file for time trend regression",
        type=["csv"],
        key="time_trend_file"
    )

    remove_outliers_time = st.checkbox("Remove outliers from value before time trend regression")

    if time_file is not None:
        try:
            df = pd.read_csv(time_file)

            st.write("Preview of uploaded file:")
            st.write("First 5 rows:")
            st.dataframe(df.head())

            st.write("Last 5 rows:")
            st.dataframe(df.tail())

            required_cols = ["date", "value"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                if st.button("Run Time Trend Regression"):
                    reg_df = df.copy()

                    reg_df["date"] = pd.to_datetime(reg_df["date"], errors="coerce")
                    reg_df["value"] = pd.to_numeric(reg_df["value"], errors="coerce")
                    reg_df = reg_df.dropna(subset=["date", "value"]).copy()
                    reg_df = reg_df.sort_values("date").reset_index(drop=True)

                    before_rows = len(reg_df)

                    if remove_outliers_time:
                        reg_df = remove_outliers_iqr(reg_df, ["value"])

                    after_rows = len(reg_df)

                    if len(reg_df) < 2:
                        st.error("Not enough valid rows to run regression.")
                    else:
                        reg_df["time_numeric"] = reg_df["date"].map(pd.Timestamp.toordinal)

                        X = sm.add_constant(reg_df["time_numeric"])
                        y = reg_df["value"]

                        model = sm.OLS(y, X).fit()

                        reg_df["predicted"] = model.predict(X)
                        reg_df["residual"] = reg_df["value"] - reg_df["predicted"]

                        st.subheader("Regression Summary")
                        if remove_outliers_time:
                            st.write(f"Outlier removal applied. Rows before: {before_rows}, rows after: {after_rows}")
                        st.text(model.summary())

                        slope = model.params["time_numeric"]
                        intercept = model.params["const"]
                        p_value = model.pvalues["time_numeric"]
                        r_squared = model.rsquared

                        st.subheader("Actual Values vs Trend Line")
                        fig1, ax1 = plt.subplots()
                        ax1.plot(reg_df["date"], reg_df["value"], label="Actual", marker="o")
                        ax1.plot(reg_df["date"], reg_df["predicted"], label="Trend Line", linestyle="--")
                        ax1.set_xlabel("Date")
                        ax1.set_ylabel("Value")
                        ax1.legend()
                        st.pyplot(fig1)

                        st.subheader("Residual Plot")
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(reg_df["date"], reg_df["residual"])
                        ax2.axhline(0, color="red", linestyle="--")
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel("Residual")
                        st.pyplot(fig2)

                        st.subheader("Regression Data Used")
                        st.dataframe(reg_df.head())

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------
# Linear REGRESSION TAB
# ---------------------------------------------------

with tab2:
    st.subheader("Linear Regression")
    st.write(
        "Upload two cleaned files with columns `date` and `value`. "
        "The independent file provides X values and the dependent file provides Y values."
    )

    ind_file = st.file_uploader(
        "Upload Independent Variable File",
        type=["csv"],
        key=f"ind_file_{st.session_state['regression_uploader_key']}"
    )

    dep_file = st.file_uploader(
        "Upload Dependent Variable File",
        type=["csv"],
        key=f"dep_file_{st.session_state['regression_uploader_key']}"
    )

    analysis_mode = st.radio(
        "Choose analysis mode",
        ["All Years Combined", "Year-by-Year"],
        key="linear_analysis_mode"
    )

    remove_outliers_linear = st.checkbox(
    "Remove outliers before linear regression"
    )

    if ind_file is not None:
        st.session_state["ind_df"] = pd.read_csv(ind_file)

    if dep_file is not None:
        st.session_state["dep_df"] = pd.read_csv(dep_file)

    if "ind_df" in st.session_state and "dep_df" in st.session_state:
        try:
            ind_df = st.session_state["ind_df"].copy()
            dep_df = st.session_state["dep_df"].copy()

            st.write("Independent file preview:")
            st.write("First 5 rows:")
            st.dataframe(ind_df.head())

            st.write("Dependent file preview:")
            st.write("First 5 rows:")
            st.dataframe(dep_df.head())

            required_cols = ["date", "value"]

            missing_ind = [col for col in required_cols if col not in ind_df.columns]
            missing_dep = [col for col in required_cols if col not in dep_df.columns]

            if missing_ind:
                st.error(f"Independent file is missing columns: {missing_ind}")
            elif missing_dep:
                st.error(f"Dependent file is missing columns: {missing_dep}")
            else:
                if st.button("Run Linear Regression"):
                    # Prepare independent data
                    ind_df["date"] = pd.to_datetime(ind_df["date"], errors="coerce")
                    ind_df["value"] = pd.to_numeric(ind_df["value"], errors="coerce")
                    ind_df = ind_df.dropna(subset=["date", "value"]).copy()
                    ind_df = ind_df.rename(columns={"value": "x"})

                    # Prepare dependent data
                    dep_df["date"] = pd.to_datetime(dep_df["date"], errors="coerce")
                    dep_df["value"] = pd.to_numeric(dep_df["value"], errors="coerce")
                    dep_df = dep_df.dropna(subset=["date", "value"]).copy()
                    dep_df = dep_df.rename(columns={"value": "y"})

                    # Merge
                    merged_df = pd.merge(
                        ind_df[["date", "x"]],
                        dep_df[["date", "y"]],
                        on="date",
                        how="inner"
                    ).sort_values("date").reset_index(drop=True)

                    if len(merged_df) < 2:
                        st.error("Not enough matching dates between the two files to run regression.")
                    else:
                        merged_df["year"] = merged_df["date"].dt.year

                        st.subheader("Merged Data Preview")
                        st.write("First 5 rows:")
                        st.dataframe(merged_df.head())

                        st.write("Last 5 rows:")
                        st.dataframe(merged_df.tail())

                        st.write(f"Matched rows: {len(merged_df)}")

                        # -----------------------------------
                        # ALL YEARS COMBINED
                        # -----------------------------------
                        if analysis_mode == "All Years Combined":

                            before_rows = len(merged_df)

                            if remove_outliers_linear:
                                merged_df = remove_outliers_iqr(merged_df, ["x", "y"])

                            after_rows = len(merged_df)

                            if remove_outliers_linear:
                                st.write(f"Outlier removal applied. Rows before: {before_rows}, rows after: {after_rows}")

                            X = sm.add_constant(merged_df["x"])
                            y = merged_df["y"]

                            model = sm.OLS(y, X).fit()

                            merged_df["predicted"] = model.predict(X)
                            merged_df["residual"] = merged_df["y"] - merged_df["predicted"]

                            st.subheader("Regression Summary")
                            st.text(model.summary())

                            st.subheader("Key Outputs")
                            slope = model.params["x"]
                            intercept = model.params["const"]
                            p_value = model.pvalues["x"]
                            r_squared = model.rsquared

                            st.write(f"Intercept: **{intercept:.6f}**")
                            st.write(f"Slope: **{slope:.6f}**")
                            st.write(f"P-value: **{p_value:.6f}**")
                            st.write(f"R-squared: **{r_squared:.4f}**")

                            st.subheader("What This Means")
                            st.write(
                                "The slope measures the expected change in the dependent variable "
                                "for a one-unit increase in the independent variable across all years combined."
                            )

                            st.subheader("Scatter Plot with Regression Line")
                            fig3, ax3 = plt.subplots()
                            ax3.scatter(merged_df["x"], merged_df["y"], label="Observed")
                            ax3.plot(merged_df["x"], merged_df["predicted"], color="red", label="Regression Line")
                            ax3.set_xlabel("Independent Variable (x)")
                            ax3.set_ylabel("Dependent Variable (y)")
                            ax3.legend()
                            st.pyplot(fig3)

                            st.subheader("Residual Plot")
                            fig4, ax4 = plt.subplots()
                            ax4.scatter(merged_df["x"], merged_df["residual"])
                            ax4.axhline(0, color="red", linestyle="--")
                            ax4.set_xlabel("Independent Variable (x)")
                            ax4.set_ylabel("Residual")
                            st.pyplot(fig4)

                        # -----------------------------------
                        # YEAR-BY-YEAR
                        # -----------------------------------
                        elif analysis_mode == "Year-by-Year":
                            results = []

                            for year, year_df in merged_df.groupby("year"):
                                year_df = year_df.copy()
                                before_rows_year = len(year_df)

                                if remove_outliers_linear:
                                    year_df = remove_outliers_iqr(year_df, ["x", "y"])

                                after_rows_year = len(year_df)

                                if len(year_df) < 2:
                                    continue

                                X_year = sm.add_constant(year_df["x"])
                                y_year = year_df["y"]

                                model_year = sm.OLS(y_year, X_year).fit()

                                results.append({
                                    "Year": year,
                                    "Observations": len(year_df),
                                    "Intercept": model_year.params.get("const", None),
                                    "Slope": model_year.params.get("x", None),
                                    "P-value": model_year.pvalues.get("x", None),
                                    "R-squared": model_year.rsquared
                                })

                            if not results:
                                st.error("No individual years had enough data to run regression.")
                            else:
                                results_df = pd.DataFrame(results)

                                st.subheader("Year-by-Year Regression Results")
                                st.dataframe(results_df)

                                selected_year = st.selectbox(
                                    "Select a year to view detailed regression output",
                                    options=results_df["Year"].tolist()
                                )

                                selected_df = merged_df[merged_df["year"] == selected_year].copy()

                                X_sel = sm.add_constant(selected_df["x"])
                                y_sel = selected_df["y"]

                                selected_model = sm.OLS(y_sel, X_sel).fit()

                                selected_df["predicted"] = selected_model.predict(X_sel)
                                selected_df["residual"] = selected_df["y"] - selected_df["predicted"]

                                st.subheader(f"Detailed Regression Summary for {selected_year}")
                                st.text(selected_model.summary())

                                st.subheader(f"Scatter Plot with Regression Line for {selected_year}")
                                fig5, ax5 = plt.subplots()
                                ax5.scatter(selected_df["x"], selected_df["y"], label="Observed")
                                ax5.plot(selected_df["x"], selected_df["predicted"], color="red", label="Regression Line")
                                ax5.set_xlabel("Independent Variable (x)")
                                ax5.set_ylabel("Dependent Variable (y)")
                                ax5.legend()
                                st.pyplot(fig5)

                                st.subheader(f"Residual Plot for {selected_year}")
                                fig6, ax6 = plt.subplots()
                                ax6.scatter(selected_df["x"], selected_df["residual"])
                                ax6.axhline(0, color="red", linestyle="--")
                                ax6.set_xlabel("Independent Variable (x)")
                                ax6.set_ylabel("Residual")
                                st.pyplot(fig6)

        except Exception as e:
            st.error(f"Error: {e}")

st.write("---")
if st.button("Clear All"):
    clear_regression_state()