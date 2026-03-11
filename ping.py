import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Define features and target
X = df.drop(columns=['fake'])
y = df['fake']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# Streamlit app
st.title("Fake vs Real Account Prediction")
st.write("Enter account details to predict if it's fake or real.")

# User inputs
profile_pic = st.selectbox("Profile Picture Present?", [0, 1])
nums_length_username = st.number_input("Numbers/Length of Username", min_value=0.0, max_value=1.0, step=0.01)
fullname_words = st.number_input("Fullname Words", min_value=0, step=1)
nums_length_fullname = st.number_input("Numbers/Length of Fullname", min_value=0.0, max_value=1.0, step=0.01)
name_equals_username = st.selectbox("Name equals Username?", [0, 1])
description_length = st.number_input("Description Length", min_value=0, step=1)
external_URL = st.selectbox("External URL Present?", [0, 1])
private = st.selectbox("Private Account?", [0, 1])
posts = st.number_input("Number of Posts", min_value=0, step=1)
followers = st.number_input("Number of Followers", min_value=0, step=1)
follows = st.number_input("Number of Follows", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    model = load_model()
    features = np.array([[profile_pic, nums_length_username, fullname_words, nums_length_fullname,
                          name_equals_username, description_length, external_URL, private,
                          posts, followers, follows]])
    prediction = model.predict(features)
    result = "Fake" if prediction[0] == 1 else "Real"
    st.write(f"The account is predicted to be: **{result}**")
    
    # If the account is fake, show report option
    if result == "Fake":
        st.write("### Report this Fake Account")
        report_text = st.text_area("Enter details for reporting:")
        if st.button("Submit Report"):
            st.success("Report submitted successfully!")
            st.experimental_rerun()
