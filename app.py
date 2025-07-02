import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load('crop_recommendation_model.pkl')

# Load the district-wise rainfall data
district_rainfall_df = pd.read_parquet('district_wise_rainfall_normal.parquet')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Function to display the homepage
def show_homepage():
    st.title("Smart Crop Recommendation System Using ML: District and Month-Specific Insights for Optimized Agricultural Practices")
    st.header("Project Overview")
    st.write("""
       This project aims to develop a machine learning-based Smart Crop Recommendation System specifically designed for Indian farmers. By analyzing key environmental parameters such as temperature, humidity, soil pH, and rainfall, the system provides district- and month-specific crop recommendations. Leveraging supervised learning algorithms, it ranks the most suitable crops based on the given conditions to help farmers make informed decisions. The primary goal is to enhance agricultural productivity and promote sustainable farming practices by offering data-driven insights tailored to diverse regions across India.
    """)
    st.header("Project Steps")
    st.write("""
        1. **Data Collection**: Gather data on environmental factors such as temperature, humidity, pH, and rainfall for various districts and months.
        2. **Data Preprocessing**: Clean and preprocess the data to ensure it is suitable for training the machine learning model.
        3. **Model Training**: Train a machine learning model using the preprocessed data to predict the best crops to plant based on the input conditions.
        4. **Model Evaluation**: Evaluate the model's performance using appropriate metrics to ensure its accuracy and reliability.
        5. **Web Interface Development**: Develop a user-friendly web interface using Streamlit to allow farmers to input their data and get crop recommendations.
    """)
    st.header("How to Use")
    st.write("""
        1. Click the "Go to Crop Recommendation" button below.
        2. Select your district and month.
        3. Input the values for Nitrogen (N), Phosphorus (P), Potassium (K), pH, temperature, and humidity.
        4. Click the "Predict" button to get the top crop recommendations based on your input.
    """)
    if st.button("Go to Crop Recommendation"):
        st.session_state.page = "Crop Recommendation"

# Function to display the crop recommendation page
def show_crop_recommendation():
    st.title("Crop Recommendation")

    # District selection
    district = st.selectbox("Select District", district_rainfall_df['DISTRICT'].unique())

    # Month selection
    month = st.selectbox("Select Month", ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])

    # Auto-fill rainfall based on selected district and month
    district_data = district_rainfall_df[district_rainfall_df['DISTRICT'] == district].iloc[0]
    rainfall = district_data[month]

    # Display auto-filled rainfall
    st.write(f"Rainfall (mm): {rainfall}")

    # Input fields for N, P, K, pH, temperature, and humidity
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

    # Predict button
    if st.button("Predict"):
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        # Make prediction
        prediction_probs = model.predict_proba(input_data)
        prediction_classes = model.classes_
        
        # Create a DataFrame for the prediction probabilities
        prediction_df = pd.DataFrame(prediction_probs, columns=label_encoder.inverse_transform(prediction_classes))
        
        # Get the top N recommendations
        top_n = 5  # You can change this value to get more or fewer recommendations
        top_recommendations = prediction_df.T.sort_values(by=0, ascending=False).head(top_n)
        
        # Display the top N recommendations
        st.write("Top Crop Recommendations:")
        for i, (crop, score) in enumerate(top_recommendations.iterrows(), start=1):
            st.write(f"{i}. {crop} (Confidence: {score.values[0] * 100:.2f}%)")

        # --- Add bar graph using matplotlib ---
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots()
        sns.barplot(x=top_recommendations.values.flatten() * 100,
                    y=top_recommendations.index,
                    palette='viridis',
                    ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Crop")
        ax.set_title("Top 5 Crop Predictions (Confidence Scores)")
        ax.set_xlim(0, 100)

        for i, v in enumerate(top_recommendations.values.flatten() * 100):
            ax.text(v + 1, i, f"{v:.2f}%", va='center')

        st.pyplot(fig)

    if st.button("Back to Home"):
        st.session_state.page = "Home"

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Display the appropriate page based on the session state
if st.session_state.page == "Home":
    show_homepage()
elif st.session_state.page == "Crop Recommendation":
    show_crop_recommendation()