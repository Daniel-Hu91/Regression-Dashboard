import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regression Analysis", layout="wide")
st.title("Regression Analysis")

if "regression_uploader_key" not in st.session_state:
    st.session_state["regression_uploader_key"] = 0


# -----------------------------
# HELPERS
# -----------------------------
def clear_regression_state():
    keys_to_delete = [
        "ind_df",
        "dep_df",
        "ind_label",
        "dep_label",
        "linear_analysis_mode_saved",
        "linear_results_df",
        "linear_year_models_original",
        "linear_year_models_cleaned",
        "linear_year_original_data",
        "linear_year_cleaned_data",
        "linear_all_initial_df",
        "linear_all_clean_df",
        "linear_all_model",
        "linear_clean_model",
        "linear_rows_before",
        "linear_rows_after",
        "years_omit_input",
        "ind_label_input",
        "dep_label_input",
        "remove_outliers_linear",
        "residual_z_threshold",
        "remove_only_upper_outliers",
        "linear_analysis_mode",
    ]

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["years_omit_input"] = ""
    st.session_state["ind_label_input"] = ""
    st.session_state["dep_label_input"] = ""
    st.session_state["remove_outliers_linear"] = True
    st.session_state["residual_z_threshold"] = 0.5
    st.session_state["remove_only_upper_outliers"] = True
    st.session_state["linear_analysis_mode"] = "All Years Combined"

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


def remove_outliers_by_residuals(df, x_col, y_col, z_threshold=0.5, upper_only=False):
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


def fit_linear_model(df, x_col="x", y_col="y"):
    X = sm.add_constant(df[[x_col]])
    y = df[y_col]
    model = sm.OLS(y, X).fit()

    out = df.copy()
    out["predicted"] = model.predict(X)
    out["residual"] = out[y_col] - out["predicted"]
    return model, out


def make_scatter_plot(df, x_col, y_col, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[x_col], df[y_col], label="Observed", alpha=0.7)
    line_df = df.sort_values(x_col)
    if "predicted" in line_df.columns:
        ax.plot(line_df[x_col], line_df["predicted"], color="red", label="Regression Line")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    return fig


def make_dual_axis_time_series(df, date_col, x_col, y_col, x_label, y_label, title):
    fig, ax1 = plt.subplots(figsize=(10, 5))

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

    time_file = st.file_uploader(
        "Upload cleaned file for time trend regression",
        type=["csv", "xlsx", "xls"],
        key=f"time_trend_file_{st.session_state['regression_uploader_key']}"
    )

    if time_file is not None:
        try:
            df = standardize_columns(load_uploaded_file(time_file))

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

                    if len(reg_df) < 2:
                        st.error("Not enough valid rows to run regression.")
                    else:
                        reg_df["time_numeric"] = reg_df["date"].map(pd.Timestamp.toordinal)

                        X = sm.add_constant(reg_df[["time_numeric"]])
                        y = reg_df["value"]

                        model = sm.OLS(y, X).fit()
                        reg_df["predicted"] = model.predict(X)
                        reg_df["residual"] = reg_df["value"] - reg_df["predicted"]

                        st.subheader("Regression Summary")
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
                        st.pyplot(fig_raw)

                        st.subheader("Residual Plot")
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(reg_df["date"], reg_df["residual"])
                        ax2.axhline(0, color="red", linestyle="--")
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel("Residual")
                        st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------
# LINEAR REGRESSION TAB
# ---------------------------------------------------
with tab2:
    st.subheader("Linear Regression")

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

    ind_label_input = st.text_input(
        "Independent variable label",
        value=st.session_state.get("ind_label_input", ""),
        key="ind_label_input"
    )

    dep_label_input = st.text_input(
        "Dependent variable label",
        value=st.session_state.get("dep_label_input", ""),
        key="dep_label_input"
    )

    analysis_mode = st.radio(
        "Choose analysis mode",
        ["All Years Combined", "Year-by-Year"],
        key="linear_analysis_mode"
    )

    years_omit_input = st.text_input(
        "Years to omit (optional: e.g. 2019 or 2020-2022 or 2019,2021-2023)",
        value=st.session_state.get("years_omit_input", ""),
        key="years_omit_input"
    )

    remove_outliers_linear = st.checkbox(
        "Remove outliers using residuals before linear regression",
        value=st.session_state.get("remove_outliers_linear", True),
        key="remove_outliers_linear"
    )

    residual_z_threshold = st.number_input(
        "Residual z-score threshold for outlier removal",
        min_value=0.1,
        value=st.session_state.get("residual_z_threshold", 0.5),
        step=0.1,
        key="residual_z_threshold"
    )

    remove_only_upper_outliers = st.checkbox(
        "Remove only upper residual outliers",
        value=st.session_state.get("remove_only_upper_outliers", True),
        key="remove_only_upper_outliers"
    )

    if ind_file is not None:
        st.session_state["ind_df"] = standardize_columns(load_uploaded_file(ind_file))

    if dep_file is not None:
        st.session_state["dep_df"] = standardize_columns(load_uploaded_file(dep_file))

    st.session_state["ind_label"] = ind_label_input
    st.session_state["dep_label"] = dep_label_input

    if "ind_df" in st.session_state and "dep_df" in st.session_state:
        try:
            ind_df = st.session_state["ind_df"].copy()
            dep_df = st.session_state["dep_df"].copy()

            ind_label = st.session_state.get("ind_label", "") or "Independent Variable"
            dep_label = st.session_state.get("dep_label", "") or "Dependent Variable"

            st.write("Independent file preview:")
            st.dataframe(ind_df.head())

            st.write("Dependent file preview:")
            st.dataframe(dep_df.head())

            required_cols = ["date", "value"]
            missing_ind = [col for col in required_cols if col not in ind_df.columns]
            missing_dep = [col for col in required_cols if col not in dep_df.columns]

            if missing_ind:
                st.error(f"Independent file is missing columns: {missing_ind}")
            elif missing_dep:
                st.error(f"Dependent file is missing columns: {missing_dep}")
            else:
                run_clicked = st.button("Run Linear Regression")

                if run_clicked:
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

                        if analysis_mode == "All Years Combined":
                            df_all = merged_df.copy()
                            st.session_state["linear_rows_before"] = len(df_all)

                            initial_model, df_all_fitted = fit_linear_model(df_all, x_col="x", y_col="y")

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

                            st.session_state["linear_rows_after"] = len(df_clean)

                            clean_model, df_clean_fitted = fit_linear_model(df_clean, x_col="x", y_col="y")

                            st.session_state["linear_all_initial_df"] = df_all_fitted
                            st.session_state["linear_all_clean_df"] = df_clean_fitted
                            st.session_state["linear_all_model"] = initial_model
                            st.session_state["linear_clean_model"] = clean_model

                        elif analysis_mode == "Year-by-Year":
                            results = []
                            year_models_original = {}
                            year_models_cleaned = {}
                            year_original_data = {}
                            year_cleaned_data = {}

                            for year, year_df in merged_df.groupby("year"):
                                year_df = year_df.copy()
                                before_year_rows = len(year_df)

                                if len(year_df) < 2:
                                    continue

                                original_model, year_df_original = fit_linear_model(year_df, x_col="x", y_col="y")

                                if remove_outliers_linear and len(year_df) >= 3:
                                    year_df_clean = remove_outliers_by_residuals(
                                        year_df,
                                        x_col="x",
                                        y_col="y",
                                        z_threshold=residual_z_threshold,
                                        upper_only=remove_only_upper_outliers
                                    )
                                else:
                                    year_df_clean = year_df.copy()

                                after_year_rows = len(year_df_clean)

                                if len(year_df_clean) < 2:
                                    continue

                                cleaned_model, year_df_clean_fitted = fit_linear_model(year_df_clean, x_col="x", y_col="y")

                                results.append({
                                    "Year": year,
                                    "Observations Before Filtering": before_year_rows,
                                    "Observations After Filtering": after_year_rows,
                                    "Rows Removed": before_year_rows - after_year_rows,
                                    "Intercept Original": original_model.params.get("const", None),
                                    "Slope Original": original_model.params.get("x", None),
                                    "P-value Original": original_model.pvalues.get("x", None),
                                    "R-squared Original": original_model.rsquared,
                                    "Intercept Cleaned": cleaned_model.params.get("const", None),
                                    "Slope Cleaned": cleaned_model.params.get("x", None),
                                    "P-value Cleaned": cleaned_model.pvalues.get("x", None),
                                    "R-squared Cleaned": cleaned_model.rsquared,
                                })

                                year_models_original[year] = original_model
                                year_models_cleaned[year] = cleaned_model
                                year_original_data[year] = year_df_original
                                year_cleaned_data[year] = year_df_clean_fitted

                            if not results:
                                st.error("No individual years had enough data to run regression.")
                            else:
                                st.session_state["linear_results_df"] = pd.DataFrame(results)
                                st.session_state["linear_year_models_original"] = year_models_original
                                st.session_state["linear_year_models_cleaned"] = year_models_cleaned
                                st.session_state["linear_year_original_data"] = year_original_data
                                st.session_state["linear_year_cleaned_data"] = year_cleaned_data

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            if "linear_analysis_mode_saved" in st.session_state:
                saved_mode = st.session_state["linear_analysis_mode_saved"]

                if saved_mode == "All Years Combined" and "linear_all_clean_df" in st.session_state:
                    df_all = st.session_state["linear_all_initial_df"].copy()
                    df_clean = st.session_state["linear_all_clean_df"].copy()
                    initial_model = st.session_state["linear_all_model"]
                    clean_model = st.session_state["linear_clean_model"]

                    st.subheader("Merged Data Preview")
                    st.write("Original first 5 rows:")
                    st.dataframe(df_all.head())

                    st.write("Cleaned first 5 rows:")
                    st.dataframe(df_clean.head())

                    if years_omit_input.strip():
                        st.write(f"Years omitted: {parse_years_to_omit(years_omit_input)}")

                    st.write(f"Rows before filtering: {st.session_state['linear_rows_before']}")
                    st.write(f"Rows after filtering: {st.session_state['linear_rows_after']}")
                    st.write(f"Rows removed: {st.session_state['linear_rows_before'] - st.session_state['linear_rows_after']}")

                    st.subheader("Original Regression Summary")
                    st.text(initial_model.summary())

                    st.subheader("Original Scatter Plot with Regression Line")
                    fig_orig = make_scatter_plot(
                        df_all, "x", "y", ind_label, dep_label, "All Years Combined - Original"
                    )
                    st.pyplot(fig_orig)

                    st.subheader("Original Time Series")
                    fig_ts_orig = make_dual_axis_time_series(
                        df_all.sort_values("date"),
                        "date",
                        "x",
                        "y",
                        ind_label,
                        dep_label,
                        "All Years Combined - Original Time Series"
                    )
                    st.pyplot(fig_ts_orig)

                    st.subheader("Cleaned Regression Summary")
                    st.text(clean_model.summary())

                    st.subheader("Cleaned Scatter Plot with Regression Line")
                    fig_clean = make_scatter_plot(
                        df_clean, "x", "y", ind_label, dep_label, "All Years Combined - Cleaned"
                    )
                    st.pyplot(fig_clean)

                    st.subheader("Cleaned Time Series")
                    fig_ts_clean = make_dual_axis_time_series(
                        df_clean.sort_values("date"),
                        "date",
                        "x",
                        "y",
                        ind_label,
                        dep_label,
                        "All Years Combined - Cleaned Time Series"
                    )
                    st.pyplot(fig_ts_clean)

                    st.subheader("Residual Plot After Filtering")
                    fig4, ax4 = plt.subplots()
                    ax4.scatter(df_clean["predicted"], df_clean["residual"])
                    ax4.axhline(0, color="red", linestyle="--")
                    ax4.set_xlabel("Fitted Values")
                    ax4.set_ylabel("Residuals")
                    st.pyplot(fig4)

                elif saved_mode == "Year-by-Year" and "linear_results_df" in st.session_state:
                    results_df = st.session_state["linear_results_df"]
                    year_models_original = st.session_state["linear_year_models_original"]
                    year_models_cleaned = st.session_state["linear_year_models_cleaned"]
                    year_original_data = st.session_state["linear_year_original_data"]
                    year_cleaned_data = st.session_state["linear_year_cleaned_data"]

                    st.subheader("Year-by-Year Regression Results Table")
                    st.dataframe(results_df)

                    st.markdown("---")
                    st.subheader("All Original Year Graphs")

                    for year in sorted(year_original_data.keys()):
                        st.markdown(f"### Original - {year}")
                        st.text(year_models_original[year].summary())

                        fig_year_orig = make_scatter_plot(
                            year_original_data[year],
                            "x",
                            "y",
                            ind_label,
                            dep_label,
                            f"{year} Original"
                        )
                        st.pyplot(fig_year_orig)

                        fig_year_ts_orig = make_dual_axis_time_series(
                            year_original_data[year].sort_values("date"),
                            "date",
                            "x",
                            "y",
                            ind_label,
                            dep_label,
                            f"{year} Original Time Series"
                        )
                        st.pyplot(fig_year_ts_orig)

                    st.markdown("---")
                    st.subheader("All Cleaned Year Graphs")

                    for year in sorted(year_cleaned_data.keys()):
                        st.markdown(f"### Cleaned - {year}")
                        st.text(year_models_cleaned[year].summary())

                        fig_year_clean = make_scatter_plot(
                            year_cleaned_data[year],
                            "x",
                            "y",
                            ind_label,
                            dep_label,
                            f"{year} Cleaned"
                        )
                        st.pyplot(fig_year_clean)

                        fig_year_ts_clean = make_dual_axis_time_series(
                            year_cleaned_data[year].sort_values("date"),
                            "date",
                            "x",
                            "y",
                            ind_label,
                            dep_label,
                            f"{year} Cleaned Time Series"
                        )
                        st.pyplot(fig_year_ts_clean)

        except Exception as e:
            st.error(f"Error: {e}")

st.write("---")
if st.button("Clear All"):
    clear_regression_state()