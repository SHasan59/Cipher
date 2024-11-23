import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Cipher",
    page_icon="🌍",
    layout="wide",  # Use a wide layout for better space utilization
)

# Add a custom style
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f2f6; /* Soft background */
        }
        .main-header {
            font-size: 3rem;
            color: #2c3e50; /* Dark slate gray */
            text-align: center;
            margin-top: 20px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: #95a5a6; /* Light gray */
            padding: 10px 0;
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Header with emoji
st.markdown('<h1 class="main-header">🌍 Cipher</h1>', unsafe_allow_html=True)



st.markdown(
    """
    <div style="text-align: center;">
        <p style="font-size: 1.2rem;">Cipher is a cutting-edge data science project that uses machine learning to detect anomalies in network traffic and prevent DDoS attacks, ensuring network security.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# REMOVE THIS CODE JUST USED FOR DEMO



# SAMPLE CODE MADE FOR DF
# Add some example content
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
    <div>
    """,
    unsafe_allow_html=True,
)
data = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["Feature A", "Feature B", "Feature C"]
)
st.dataframe(data, use_container_width=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# SAMPLE CODE FOR FILE UPLOAD
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt", "json"])

if uploaded_file is not None:
    # Display the uploaded file's name
    st.write(f"File uploaded: {uploaded_file.name}")

    # Read and display the content of the file based on type
    if uploaded_file.type == "text/csv":
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write(df)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        import pandas as pd
        df = pd.read_excel(uploaded_file)
        st.write(df)
    elif uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8")
        st.text(file_content)
    elif uploaded_file.type == "application/json":
        import json
        file_content = json.load(uploaded_file)
        st.json(file_content)