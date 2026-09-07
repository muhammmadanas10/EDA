import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Interface", layout="wide")

st.title("Exploratory Data Analysis Interface")

st.sidebar.title("Dataset Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            st.error("Validation Error: The uploaded CSV file contains no data.")
        else:
            st.sidebar.subheader("Attribute Selection")
            selected_col = st.sidebar.selectbox("Select Attribute for Visualization:", df.columns)

            st.header("Dataset Preview & Metadata")
            
            st.write("First 5 Rows:")
            st.dataframe(df.head(), use_container_width=True)

            st.write("Shape:", df.shape)

            st.write("Column Data Types:")
            dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"])
            st.dataframe(dtypes_df, use_container_width=True)

            st.write("Missing Values per Column:")
            missing_df = pd.DataFrame({
                "Missing Count": df.isnull().sum(),
                "Missing (%)": (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)

            st.write("Statistical Summary (Numerical Attributes):")
            num_df = df.select_dtypes(include=["number"])
            if not num_df.empty:
                stats_df = pd.DataFrame({
                    "Mean": num_df.mean(),
                    "Median": num_df.median(),
                    "Min": num_df.min(),
                    "Max": num_df.max()
                })
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No numerical attributes available.")

            st.markdown("---")
            st.header("Visualization")

            if selected_col:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                    ax.set_title("Histogram of " + str(selected_col))
                    ax.set_xlabel(str(selected_col))
                    ax.set_ylabel("Frequency")
                else:
                    value_counts = df[selected_col].value_counts()
                    ax.bar(value_counts.index.astype(str), value_counts.values)
                    ax.set_title("Bar Chart of " + str(selected_col))
                    ax.set_xlabel(str(selected_col))
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)

                st.pyplot(fig)

    except pd.errors.EmptyDataError:
        st.error("Validation Error: The file is empty.")
    except pd.errors.ParserError:
        st.error("Validation Error: Failed to parse CSV file.")
    except Exception as e:
        st.error("An unexpected error occurred: " + str(e))