import streamlit as st
import pandas as pd
from utils import (
    signup_page, login_page, start_webcam_for_waste_detection,
    user_waste_analysis, overall_waste_trends, predict_future_waste_trends
)

# Initialize session state for user
if "user" not in st.session_state or st.session_state.user is None:
    st.session_state.user = None

if st.session_state.user:
    st.title(f"Welcome, {st.session_state.user['name']}!")
    st.sidebar.write(f"Account Number: {st.session_state.user['account_number']}")

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.success("Logged out successfully!")

    st.header("Dashboard Actions")
    if st.button("Grab the Waste"):
        start_webcam_for_waste_detection()

    st.header("Waste Data Analysis")

    # Ensure user object has required fields before proceeding
    if isinstance(st.session_state.user,
                  dict) and "account_number" in st.session_state.user and "email" in st.session_state.user:
        user_waste_analysis(st.session_state.user["account_number"], st.session_state.user["email"])
        overall_waste_trends()
        predict_future_waste_trends(st.session_state.user["account_number"], st.session_state.user["email"])
    else:
        st.warning("User session not properly initialized. Please log in again.")

else:
    # Login or Signup page navigation
    page = st.sidebar.radio("Choose a page", ["Login", "Signup"])
    if page == "Signup":
        signup_page()
    else:
        login_page()
