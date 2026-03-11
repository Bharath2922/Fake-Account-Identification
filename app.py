import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from auth import login_page, register_page, logout
import time

# Session and auth state
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["home"])[0]

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    if current_page == "register":
        register_page()
    else:
        login_page()
    st.stop()

# Logout button
st.sidebar.button("Logout", on_click=logout)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Dashboard", ["Home", "Prediction", "Contact Us"])

# Custom CSS
def set_custom_styles(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('{image_url}') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Arial', sans-serif;
            color: #f5f6fa;
        }}
        .sidebar .sidebar-content {{
            background-color: #2f3640;
            color: black;
        }}
        .sidebar .sidebar-content .css-1d391kg {{
            color: black;
        }}
        .stButton>button {{
            background-color: #e84118;
            color: white;
            border-radius: 5px;
        }}
        .stButton>button:hover {{
            background-color: #c23616;
        }}
        select {{
            background-color: #2f3640;
            color: black;
            padding: 10px;
            border-radius: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Home Page
if page == "Home":
    set_custom_styles("https://img.freepik.com/free-vector/blue-social-media-background_1017-7008.jpg")
    st.title("Welcome to Fake Account Detection System")
    st.write("This interactive tool helps identify fake social media accounts using machine learning techniques.")
    st.write("Navigate to the *Prediction* page to check if an account is fake or real.")

# Contact Us Page
elif page == "Contact Us":
    set_custom_styles("https://img.freepik.com/free-vector/blue-social-media-background_1017-7008.jpg")
    st.title("Contact Us")
    st.write("Have questions? Send us a message below:")
    st.write(" **Email:** Serversites@gmail.com")
    st.write(" **Website:** [www.socialserver.com](http://www.socialserver.com)")

    message = st.text_area("Your Message")

    if st.button("Send Message"):
        if message.strip():
            popup_placeholder = st.empty()
            popup_placeholder.success(" Message sent successfully!", icon="✅")
            time.sleep(2)
            popup_placeholder.empty()
            st.write("### 🎉 **Thank you for your feedback!**")
            st.write("We will keep you updated.")
            st.rerun()
        else:
            st.warning(" Please enter a message before submitting.")

# Prediction Page
elif page == "Prediction":
    set_custom_styles("https://img.freepik.com/free-vector/blue-social-media-background_1017-7008.jpg")

    st.subheader("User Details")
    st.write(f"👤 **Username:** {st.session_state.username}")

    # Load and train model
    file_path = "train.csv"
    df = pd.read_csv(file_path)

    X = df.drop(columns=['fake'])
    y = df['fake']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    def load_model():
        with open("model.pkl", "rb") as f:
            return pickle.load(f)

    # Input fields
    st.title("Fake vs Real Account Prediction")
    st.write("Fill in the details below to predict if an account is *Fake* or *Real*.")

    profile_pic = st.selectbox("Profile Picture Present?", ["No", "Yes"])
    nums_length_username = st.slider("Numbers/Length of Username", min_value=0.0, max_value=1.0, step=0.01)
    fullname_words = st.number_input("Fullname Words", min_value=0, step=1)
    nums_length_fullname = st.slider("Numbers/Length of Fullname", min_value=0.0, max_value=1.0, step=0.01)
    name_equals_username = st.selectbox("Name Equals Username?", ["No", "Yes"])
    description_length = st.number_input("Description Length", min_value=0, step=1)
    external_URL = st.selectbox("External URL Present?", ["No", "Yes"])
    private = st.selectbox("Private Account?", ["No", "Yes"])
    posts = st.number_input("Number of Posts", min_value=0, step=1)
    followers = st.number_input("Number of Followers", min_value=0, step=1)
    follows = st.number_input("Number of Follows", min_value=0, step=1)

    # Prediction
    if st.button("Predict"):
        model = load_model()
        features = np.array([[1 if profile_pic == "Yes" else 0,
                              nums_length_username, fullname_words, nums_length_fullname,
                              1 if name_equals_username == "Yes" else 0,
                              description_length, 1 if external_URL == "Yes" else 0,
                              1 if private == "Yes" else 0, posts, followers, follows]])
        prediction = model.predict(features)
        result = "Fake" if prediction[0] == 1 else "Real"

        st.success(f"The account is predicted to be: *{result}*")

        # Accuracy Display
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy score
        st.write(f" **Model Accuracy:** {accuracy * 100:.2f}%")

        # Summary
        st.write("### User Input Summary:")
        st.write(f"👤 **Username:** {st.session_state.username}")
        st.write(f" **Profile Picture Present:** {profile_pic}")
        st.write(f" **Numbers/Length of Username:** {nums_length_username}")
        st.write(f" **Fullname Words:** {fullname_words}")
        st.write(f" **Numbers/Length of Fullname:** {nums_length_fullname}")
        st.write(f" **Name Equals Username:** {name_equals_username}")
        st.write(f" **Description Length:** {description_length}")
        st.write(f" **External URL Present:** {external_URL}")
        st.write(f" **Private Account:** {private}")
        st.write(f" **Number of Posts:** {posts}")
        st.write(f" **Number of Followers:** {followers}")
        st.write(f" **Number of Follows:** {follows}")

        # Report section
        if result == "Fake":
            st.write("### Report this Fake Account")
            report_text = st.text_area("Enter details for reporting:")

            if "report_step" not in st.session_state:
                st.session_state.report_step = 0

            if st.button("Submit Report") and st.session_state.report_step == 0:
                st.session_state.report_step = 1
                st.success("Reported successfully!")
                st.write("Account Reported!")
                st.rerun()
