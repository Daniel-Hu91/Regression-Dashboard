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
        "linear_regression_results",
        "ind_df",
        "dep_df",
        "linear_analysis_mode",
        "linear_analysis_mode_saved",
        "remove_outliers_time",
        "remove_outliers_linear",
        "linear_merged_df",
        "linear_results_df",
        "linear_year_data",
        "selected_year",
        "linear_outlier_counts",
        "residual_z_threshold",
        "remove_only_upper_outliers",
        "years_omit_input",
        "time_residual_z_threshold",
        "time_remove_only_upper_outliers",
        "ind_label",
        "dep_label",
        "linear_all_initial_df",
        "linear_all_clean_df",
        "linear_rows_before",
        "linear_rows_after",
        "ind_label_input",
        "dep_label_input"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["regression_uploader_key"] += 1
    st.rerun()


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")


def standardize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


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
                raise ValueError(f"Invalid range: {part}")

            years.update(range(start_year, end_year + 1))
        else:
            years.add(int(part))

    return sorted(years)


def remove_outliers_by_residuals(df, x_col, y_col, z_threshold=2.0, upper_only=False):
    df = df.copy()

    if len(df) < 3:
        return df

    X = sm.add_constant(df[[x_col]])
    y = df[y_col]

    model = sm.OLS(y, X).fit()

    df["predicted_initial"] = model.predict(X)
    df["residual_initial"] = df[y_col] - df["predicted_initial"]

    resid_std = df["residual_initial"].std()

    if resid_std == 0 or pd.isna(resid_std):
        df["residual_z"] = 0
    else:
        df["residual_z"] = df["residual_initial"] / resid_std

    if upper_only:
        df_clean = df[df["residual_z"] <= z_threshold].copy()
    else:
        df_clean = df[df["residual_z"].abs() <= z_threshold].copy()

    return df_clean


def plot_dual_axis_timeseries(df, date_col="date", x_col="x", y_col="y",
                              x_label="Independent Variable",
                              y_label="Dependent Variable",
                              title="Time Series"):
    fig, ax1 = plt.subplots()

    ax1.plot(df[date_col], df[y_col], color="blue", label=y_label)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(y_label, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(df[date_col], df[x_col], color="green", label=x_label)
    ax2.set_ylabel(x_label, color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    plt.title(title)
    fig.tight_layout()
    return fig


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
        type=["csv", "xlsx", "xls"],
        key=f"time_trend_file_{st.session_state['regression_uploader_key']}"
    )

    remove_outliers_time = st.checkbox(
        "Remove outliers using residuals before time trend regression",
        key="remove_outliers_time"
    )

    time_residual_z_threshold = st.number_input(
        "Time trend residual z-score threshold",
        min_value=0.1,
        value=0.5,
        step=0.1,
        key="time_residual_z_threshold"
    )

    time_remove_only_upper_outliers = st.checkbox(
        "Time trend: remove only upper residual outliers",
        value=False,
        key="time_remove_only_upper_outliers"
    )

    if time_file is not None:
        try:
            df = standardize_columns(load_uploaded_file(time_file))

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

                    reg_df["time_numeric"] = reg_df["date"].map(pd.Timestamp.toordinal)

                    if remove_outliers_time:
                        reg_df = remove_outliers_by_residuals(
                            reg_df,
                            x_col="time_numeric",
                            y_col="value",
                            z_threshold=time_residual_z_threshold,
                            upper_only=time_remove_only_upper_outliers
                        )

                    after_rows = len(reg_df)

                    if len(reg_df) < 2:
                        st.error("Not enough valid rows to run regression.")
                    else:
                        X = sm.add_constant(reg_df[["time_numeric"]])
                        y = reg_df["value"]

                        model = sm.OLS(y, X).fit()

                        reg_df["predicted"] = model.predict(X)
                        reg_df["residual"] = reg_df["value"] - reg_df["predicted"]

                        st.subheader("Regression Summary")
                        if remove_outliers_time:
                            st.write(
                                f"Residual-based outlier removal applied. Rows before: {before_rows}, rows after: {after_rows}"
                            )
                        st.text(model.summary())

                        st.subheader("Actual Values vs Trend Line")
                        fig1, ax1 = plt.subplots()
                        ax1.plot(reg_df["date"], reg_df["value"], label="Actual", marker="o")
                        ax1.plot(reg_df["date"], reg_df["predicted"], label="Trend Line", linestyle="--")
                        ax1.set_xlabel("Date")
                        ax1.set_ylabel("Value")
                        ax1.legend()
                        st.pyplot(fig1)

                        st.subheader("Raw Time Series")
                        fig_raw, ax_raw = plt.subplots()
                        ax_raw.plot(reg_df["date"], reg_df["value"], color="blue")
                        ax_raw.set_xlabel("Date")
                        ax_raw.set_ylabel("Value")
                        ax_raw.set_title("Raw Time Series")
                        st.pyplot(fig_raw)

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
# LINEAR REGRESSION TAB
# ---------------------------------------------------
with tab2:
    st.subheader("Linear Regression")
    st.write(
        "Upload two cleaned files with columns `date` and `value`. "
        "The independent file provides X values and the dependent file provides Y values."
    )

    ind_file = st.file_uploader(
        "Upload Independent Variable File",
        type=["csv", "xlsx", "xls"],
        key=f"ind_file_{st.session_state['regression_uploader_key']}"
    )

    dep_file = st.file_uploader(
        "Upload Dependent Variable File",
        type=["csv", "xlsx", "xls"],
        key=f"dep_file_{st.session_state['regression_uploader_key']}"
    )

    ind_label_input = st.text_input("Independent variable label", value="")
    dep_label_input = st.text_input("Dependent variable label", value="")

    analysis_mode = st.radio(
        "Choose analysis mode",
        ["All Years Combined", "Year-by-Year"],
        key="linear_analysis_mode"
    )

    years_omit_input = st.text_input(
        "Years to omit (optional: e.g. 2019 or 2020-2022 or 2019,2021-2023)",
        key="years_omit_input"
    )

    remove_outliers_linear = st.checkbox(
        "Remove outliers using residuals before linear regression",
        value=True,
        key="remove_outliers_linear"
    )

    residual_z_threshold = st.number_input(
        "Residual z-score threshold for outlier removal",
        min_value=0.1,
        value=2.0,
        step=0.1,
        key="residual_z_threshold"
    )

    remove_only_upper_outliers = st.checkbox(
        "Remove only upper residual outliers",
        value=True,
        key="remove_only_upper_outliers"
    )

    if ind_file is not None:
        st.session_state["ind_df"] = standardize_columns(load_uploaded_file(ind_file))
        st.session_state["ind_label"] = ind_label_input

    if dep_file is not None:
        st.session_state["dep_df"] = standardize_columns(load_uploaded_file(dep_file))
        st.session_state["dep_label"] = dep_label_input

    if "ind_df" in st.session_state and "dep_df" in st.session_state:
        try:
            ind_df = st.session_state["ind_df"].copy()
            dep_df = st.session_state["dep_df"].copy()

            ind_label = st.session_state.get("ind_label", ind_label_input)
            dep_label = st.session_state.get("dep_label", dep_label_input)

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
                    ind_work = ind_df.copy()
                    dep_work = dep_df.copy()

                    ind_work["date"] = pd.to_datetime(ind_work["date"], errors="coerce")
                    ind_work["value"] = pd.to_numeric(ind_work["value"], errors="coerce")
                    ind_work = ind_work.dropna(subset=["date", "value"]).copy()
                    ind_work = ind_work.rename(columns={"value": "x"})

                    dep_work["date"] = pd.to_datetime(dep_work["date"], errors="coerce")
                    dep_work["value"] = pd.to_numeric(dep_work["value"], errors="coerce")
                    dep_work = dep_work.dropna(subset=["date", "value"]).copy()
                    dep_work = dep_work.rename(columns={"value": "y"})

                    merged_df = pd.merge(
                        ind_work[["date", "x"]],
                        dep_work[["date", "y"]],
                        on="date",
                        how="inner"
                    ).sort_values("date").reset_index(drop=True)

                    years_to_omit = parse_years_to_omit(years_omit_input)
                    if years_to_omit:
                        merged_df = merged_df[~merged_df["date"].dt.year.isin(years_to_omit)].copy()

                    if len(merged_df) < 2:
                        st.error("Not enough matching dates between the two files to run regression.")
                    else:
                        merged_df["year"] = merged_df["date"].dt.year
                        st.session_state["linear_analysis_mode_saved"] = analysis_mode

                        if analysis_mode == "Year-by-Year":
                            results = []
                            year_data_dict = {}

                            for year, year_df in merged_df.groupby("year"):
                                year_df = year_df.copy()
                                before_year_rows = len(year_df)

                                if len(year_df) < 2:
                                    continue

                                if remove_outliers_linear and len(year_df) >= 3:
                                    year_df = remove_outliers_by_residuals(
                                        year_df,
                                        x_col="x",
                                        y_col="y",
                                        z_threshold=residual_z_threshold,
                                        upper_only=remove_only_upper_outliers
                                    )

                                after_year_rows = len(year_df)

                                if len(year_df) < 2:
                                    continue

                                X_year = sm.add_constant(year_df[["x"]])
                                y_year = year_df["y"]

                                model_year = sm.OLS(y_year, X_year).fit()

                                results.append({
                                    "Year": year,
                                    "Observations Before Filtering": before_year_rows,
                                    "Observations After Filtering": after_year_rows,
                                    "Rows Removed": before_year_rows - after_year_rows,
                                    "Intercept": model_year.params.get("const", None),
                                    "Slope": model_year.params.get("x", None),
                                    "P-value": model_year.pvalues.get("x", None),
                                    "R-squared": model_year.rsquared
                                })

                                year_data_dict[year] = year_df.copy()

                            if not results:
                                st.error("No individual years had enough data to run regression.")
                            else:
                                st.session_state["linear_results_df"] = pd.DataFrame(results)
                                st.session_state["linear_year_data"] = year_data_dict

                        elif analysis_mode == "All Years Combined":
                            df_all = merged_df.copy()
                            before_rows = len(df_all)

                            if remove_outliers_linear:
                                df_clean = remove_outliers_by_residuals(
                                    df_all,
                                    x_col="x",
                                    y_col="y",
                                    z_threshold=residual_z_threshold,
                                    upper_only=remove_only_upper_outliers
                                )
                            else:
                                df_clean = df_all.copy()

                            after_rows = len(df_clean)

                            st.session_state["linear_all_initial_df"] = df_all.copy()
                            st.session_state["linear_all_clean_df"] = df_clean.copy()
                            st.session_state["linear_rows_before"] = before_rows
                            st.session_state["linear_rows_after"] = after_rows

            if "linear_analysis_mode_saved" in st.session_state:
                saved_mode = st.session_state["linear_analysis_mode_saved"]

                if saved_mode == "All Years Combined" and "linear_all_clean_df" in st.session_state:
                    df_all = st.session_state["linear_all_initial_df"].copy()
                    df_clean = st.session_state["linear_all_clean_df"].copy()

                    ind_label = st.session_state.get("ind_label", ind_label_input)
                    dep_label = st.session_state.get("dep_label", dep_label_input)

                    st.subheader("Merged Data Preview")
                    st.write("First 5 rows:")
                    st.dataframe(df_clean.head())

                    st.write("Last 5 rows:")
                    st.dataframe(df_clean.tail())

                    if years_omit_input.strip():
                        st.write(f"Years omitted: {parse_years_to_omit(years_omit_input)}")

                    st.write(f"Rows before filtering: {st.session_state['linear_rows_before']}")
                    st.write(f"Rows after filtering: {st.session_state['linear_rows_after']}")
                    st.write(f"Rows removed: {st.session_state['linear_rows_before'] - st.session_state['linear_rows_after']}")

                    if len(df_clean) < 2:
                        st.error("Not enough rows remaining after filtering to run regression.")
                    else:
                        X_clean = sm.add_constant(df_clean[["x"]])
                        y_clean = df_clean["y"]

                        clean_model = sm.OLS(y_clean, X_clean).fit()

                        df_clean["Fitted_Clean"] = clean_model.predict(X_clean)
                        df_clean["Residuals_Clean"] = y_clean - df_clean["Fitted_Clean"]

                        st.subheader("Cleaned Regression Summary")
                        st.text(clean_model.summary())

                        st.subheader("Key Outputs")
                        slope = clean_model.params["x"]
                        intercept = clean_model.params["const"]
                        p_value = clean_model.pvalues["x"]
                        r_squared = clean_model.rsquared

                        st.write(f"Intercept: **{intercept:.6f}**")
                        st.write(f"Slope: **{slope:.6f}**")
                        st.write(f"P-value: **{p_value:.6f}**")
                        st.write(f"R-squared: **{r_squared:.4f}**")

                        st.subheader("Scatter Plot with Regression Line")
                        fig3, ax3 = plt.subplots()
                        ax3.scatter(df_clean["x"], df_clean["y"], label="Observed")
                        line_df = df_clean.sort_values("x")
                        ax3.plot(line_df["x"], line_df["Fitted_Clean"], color="red", label="Regression Line")
                        ax3.set_xlabel(ind_label)
                        ax3.set_ylabel(dep_label)
                        ax3.legend()
                        st.pyplot(fig3)

                        st.subheader("Time Series of Independent and Dependent Variables")
                        fig_ts = plot_dual_axis_timeseries(
                            df_clean.sort_values("date"),
                            date_col="date",
                            x_col="x",
                            y_col="y",
                            x_label=ind_label,
                            y_label=dep_label,
                            title="All Years Combined Time Series"
                        )
                        st.pyplot(fig_ts)

                        st.subheader("Residual Plot After Filtering")
                        fig4, ax4 = plt.subplots()
                        ax4.scatter(df_clean["Fitted_Clean"], df_clean["Residuals_Clean"])
                        ax4.axhline(0, color="red", linestyle="--")
                        ax4.set_xlabel("Fitted Values")
                        ax4.set_ylabel("Residuals")
                        st.pyplot(fig4)

                elif saved_mode == "Year-by-Year" and "linear_results_df" in st.session_state:
                    results_df = st.session_state["linear_results_df"]
                    ind_label = st.session_state.get("ind_label", ind_label_input)
                    dep_label = st.session_state.get("dep_label", dep_label_input)

                    st.subheader("Year-by-Year Regression Results")
                    st.dataframe(results_df)

                    selected_year = st.selectbox(
                        "Select a year to view detailed regression output",
                        options=results_df["Year"].tolist(),
                        key="selected_year"
                    )

                    year_data_dict = st.session_state.get("linear_year_data", {})

                    if selected_year in year_data_dict:
                        selected_df = year_data_dict[selected_year].copy()

                        if len(selected_df) < 2:
                            st.error("Not enough rows for this year.")
                        else:
                            X_sel = sm.add_constant(selected_df[["x"]])
                            y_sel = selected_df["y"]

                            selected_model = sm.OLS(y_sel, X_sel).fit()

                            selected_df["predicted"] = selected_model.predict(X_sel)
                            selected_df["residual"] = selected_df["y"] - selected_df["predicted"]

                            st.subheader(f"Detailed Regression Summary for {selected_year}")
                            st.text(selected_model.summary())

                            st.subheader(f"Scatter Plot with Regression Line for {selected_year}")
                            fig5, ax5 = plt.subplots()
                            ax5.scatter(selected_df["x"], selected_df["y"], label="Observed")
                            line_df = selected_df.sort_values("x")
                            ax5.plot(line_df["x"], line_df["predicted"], color="red", label="Regression Line")
                            ax5.set_xlabel(ind_label)
                            ax5.set_ylabel(dep_label)
                            ax5.legend()
                            st.pyplot(fig5)

                            st.subheader(f"Time Series for {selected_year}")
                            fig_ts_year = plot_dual_axis_timeseries(
                                selected_df.sort_values("date"),
                                date_col="date",
                                x_col="x",
                                y_col="y",
                                x_label=ind_label,
                                y_label=dep_label,
                                title=f"Time Series for {selected_year}"
                            )
                            st.pyplot(fig_ts_year)

                            st.subheader(f"Residual Plot for {selected_year}")
                            fig6, ax6 = plt.subplots()
                            ax6.scatter(selected_df["x"], selected_df["residual"])
                            ax6.axhline(0, color="red", linestyle="--")
                            ax6.set_xlabel(ind_label)
                            ax6.set_ylabel("Residual")
                            st.pyplot(fig6)

        except Exception as e:
            st.error(f"Error: {e}")

st.write("---")
if st.button("Clear All"):
    clear_regression_state()