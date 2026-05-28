import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

API_KEY = ""your_api_key_here""
genai.configure(api_key=API_KEY)

# Find working model
working_model = None
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        if 'flash' in m.name or 'pro' in m.name:
            working_model = m.name
            break

st.write(f"Using model: {working_model}")
model = genai.GenerativeModel(working_model)

def analyse_data(df, question):
    summary = f"""
    Rows: {df.shape[0]}
    Columns: {df.shape[1]}
    Column names: {list(df.columns)}
    Missing values: {df.isnull().sum().to_dict()}
    Statistics: {df.describe().to_string()}
    First 5 rows: {df.head().to_string()}
    Question: {question}
    Please analyse and answer clearly with insights.
    """
    response = model.generate_content(summary)
    return response.text

st.set_page_config(page_title="AI Data Analysis Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analysis Agent")
st.subheader("Upload any CSV and let AI analyse it for you!")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded! {df.shape[0]} rows and {df.shape[1]} columns")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Automatic Analysis")
    if st.button("Run Auto Analysis"):
        with st.spinner("AI is analysing your data..."):
            analysis = analyse_data(df, "Give me a comprehensive analysis including key statistics, patterns, anomalies, and business insights.")
            st.markdown(analysis)
    
    st.subheader("Visualisations")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X axis", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y axis", numeric_cols, index=1)
        chart_type = st.selectbox("Select chart type", ["Scatter", "Bar", "Line", "Histogram"])
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        else:
            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Ask AI About Your Data")
    question = st.text_input("Ask any question about your dataset...")
    if st.button("Ask AI") and question:
        with st.spinner("AI is thinking..."):
            answer = analyse_data(df, question)
            st.markdown(answer)

else:
    st.info("Please upload a CSV file to get started!")