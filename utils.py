import streamlit as st
import random
from pymongo import MongoClient
from pyzbar.pyzbar import decode
import cv2
from ultralytics import YOLO
from datetime import datetime 

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["grabage"]
auth_collection = db["OAthu"]
grabage_items_collection = db["grabage_Items"]

# Generate a random 10-digit account number
def generate_account_number():
    return str(random.randint(1000000000, 9999999999))

# Signup page
def signup_page():
    st.title("Signup Page")
    email = st.text_input("Email")
    name = st.text_input("Name")
    password = st.text_input("Password", type='password')
    number = st.text_input("Phone Number")
    address = st.text_area("Address")
    city = st.text_input("City")
    state = st.text_input("State")
    country = st.text_input("Country")

    if st.button("Signup"):
        if auth_collection.find_one({"email": email}):
            st.error("Email already exists! Please use a different email.")
            return
        
        account_number = generate_account_number()
        user_data = {
            "email": email,
            "name": name,
            "password": password,
            "number": number,
            "address": address,
            "city": city,
            "state": state,
            "country": country,
            "account_number": account_number,
            "balance": 0  # Initial balance
        }
        auth_collection.insert_one(user_data)
        st.success(f"Signup Successful! Your account number is {account_number}")

# Login page
def login_page():
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        user = auth_collection.find_one({"email": email, "password": password})
        if user:
            st.session_state.user = user
            st.success(f"Welcome, {user['name']}!")
            display_user_info()
        else:
            st.error("Invalid credentials")

# Display user information
def display_user_info():
    user = st.session_state.user
    account_number = user["account_number"]
    email = user["email"]
        # Query the database to calculate the balance
    garbage_items = grabage_items_collection.find({"account_number": account_number, "email": email})
    total_balance = sum(item["amount"] for item in garbage_items)  # Sum up all the amounts

    # Display user information and calculated balance
    st.write(f"Username: {user['name']}")
    st.write(f"Account Number: {account_number}")
    st.write(f"Balance: ${total_balance}")  # Display dynamically calculated balance

    if st.button("Grab the Waste"):
        start_webcam_for_waste_detection()

# Initialize YOLO model
bottle_model = YOLO('yolov8l.pt')

import streamlit as st
import random
from pymongo import MongoClient
from pyzbar.pyzbar import decode
import cv2
from ultralytics import YOLO

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["grabage"]
auth_collection = db["OAthu"]
grabage_items_collection = db["grabage_Items"]

# Initialize YOLO model
model = YOLO('yolov8l.pt')

# COCO class mappings for waste items
WASTE_MAPPINGS = {
    39: {"name": "bottle", "has_barcode": True},  # bottle
    49: {"name": "can", "has_barcode": True},  # can (like soda cans, often with barcode)
    55: {"name": "food_container", "has_barcode": True},  # food container (takeout box, often has barcode)
    100: {"name": "soda_can", "has_barcode": True},  # soda can
    101: {"name": "juice_box", "has_barcode": True},  # juice box
    102: {"name": "snack_bag", "has_barcode": True},  # snack bag (like chips)
    103: {"name": "shampoo_bottle", "has_barcode": True},  # shampoo bottle
    104: {"name": "detergent_bottle", "has_barcode": True},  # detergent bottle
    105: {"name": "frozen_food", "has_barcode": True},  # frozen food packaging
    106: {"name": "cereal_box", "has_barcode": True},  # cereal box
}


def start_webcam_for_waste_detection():
    st.header("Waste Detection")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Create columns for displaying detection status
    col1, col2 = st.columns(2)
    status_placeholder = col1.empty()
    type_placeholder = col2.empty()

    # Flags for detection
    waste_detected = False
    waste_type = None
    barcode_detected = False
    barcode_data = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # Convert to grayscale for barcode detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect waste items
        if not waste_detected:
            results = model.predict(frame, conf=0.4)  # Lowered confidence threshold slightly
            for result in results[0].boxes:
                class_id = int(result.cls[0])
                confidence = float(result.conf[0])

                if class_id in WASTE_MAPPINGS and confidence > 0.4:
                    waste_detected = True
                    waste_type = WASTE_MAPPINGS[class_id]["name"]
                    # Draw detection
                    frame = results[0].plot()

                    # Display detection label
                    display_name = waste_type.replace('_', ' ').title()
                    cv2.putText(
                        frame,
                        f"{display_name} Detected!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Detect barcode
        if not barcode_detected:
            barcodes = decode(gray_frame)
            if barcodes:
                for barcode in barcodes:
                    (x, y, w, h) = barcode.rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    barcode_data = barcode.data.decode('utf-8')
                    barcode_detected = True
                    cv2.putText(
                        frame,
                        f"Barcode: {barcode_data}",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Update status messages
        if not waste_detected:
            status_placeholder.warning("📷 Searching for waste item...")
        else:
            display_name = waste_type.replace('_', ' ').title()
            status_placeholder.success(f"✅ {display_name} Detected!")

        if not barcode_detected:
            type_placeholder.warning("🔍 Scanning for barcode...")
        else:
            type_placeholder.success(f"✅ Barcode Found: {barcode_data}")

        # Stream the frame
        stframe.image(frame, channels="BGR", caption="Waste Detection")

        # If both waste item and barcode are detected
        if waste_detected and barcode_detected:
            # Set amount based on waste type
            amount = {
                "bottle": 2.00,
                "can": 1.50,
                "food_container": 1.00,
                "soda_can": 1.50,
                "juice_box": 1.00,
                "snack_bag": 0.75,
                "shampoo_bottle": 2.00,
                "detergent_bottle": 2.00,
                "frozen_food": 3.00,
                "cereal_box": 1.50
            }.get(waste_type, 1)  # Default to 1 if type not found

            # Add the item to MongoDB
            if st.session_state.user:
                grabage_items_collection.insert_one({
                    "account_number": st.session_state.user["account_number"],
                    "email": st.session_state.user["email"],
                    "amount": amount,
                    "barcode_data": barcode_data,
                    "item_type": waste_type,
                    "timestamp": datetime.now()
                })

                display_name = waste_type.replace('_', ' ').title()
                st.success(f"✅ {display_name} with barcode {barcode_data} added successfully!")
                st.info(f"💰 Amount credited: ${amount:.2f}")

                # Show updated balance
                show_balance(st.session_state.user["account_number"],
                             st.session_state.user["email"])

                # Close webcam after successful detection
                cap.release()
                break


def show_balance(account_number, email):
    """Display user balance"""
    garbage_items = grabage_items_collection.find({
        "account_number": account_number, 
        "email": email
    })
    
    total_balance = sum(item['amount'] for item in garbage_items)
    
    user = auth_collection.find_one({
        "account_number": account_number, 
        "email": email
    })
    
    if user:
        st.write("---")
        st.write("### Account Information")
        st.write(f"👤 Account Number: {user['account_number']}")
        st.write(f"📧 Email: {user['email']}")
        st.write(f"💰 Total Balance: ${total_balance:.2f}")
        
        # Show recent transactions
        st.write("### Recent Transactions")
        recent_items = grabage_items_collection.find({
            "account_number": account_number,
            "email": email
        }).sort("timestamp", -1).limit(5)
        
        for item in recent_items:
            # Get item type with fallback to "bottle" for older records
            display_type = item.get('item_type', 'bottle').replace('_', ' ').title()
            amount = item.get('amount', 0)
            barcode = item.get('barcode_data', 'N/A')
            
            st.write(f"- {display_type} | Barcode: {barcode} | Amount: ${amount:.2f}")
    else:
        st.error("User not found!")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["grabage"]
grabage_items_collection = db["grabage_Items"]


def overall_waste_trends():
    """Show overall waste generation trends across all users."""
    st.header("Overall Waste Generation Trends")

    waste_data = list(grabage_items_collection.find())
    if not waste_data:
        st.warning("No overall waste data available.")
        return

    df = pd.DataFrame(waste_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Overall Waste Category Distribution
    st.subheader("Total Waste Collected by Category")
    category_totals = df["item_type"].value_counts()

    # Create a bar plot with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_totals.index, y=category_totals.values, ax=ax, palette="coolwarm")

    ax.set_xlabel("Waste Category", fontsize=12, color='white')
    ax.set_ylabel("Total Count", fontsize=12, color='white')
    ax.set_title("Total Waste Collected by Category", fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xticks(rotation=45, color='white')  # Rotate x-axis labels for better visibility
    plt.yticks(color='white')  # Set y-ticks color
    fig.patch.set_facecolor('#333333')  # Dark gray background

    st.pyplot(fig)

    # Monthly Trend Analysis
    st.subheader("Monthly Waste Collection Trend")
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly_waste = df.groupby("month")["amount"].sum()

    # Create a line plot with dark theme
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_waste.index.astype(str), monthly_waste.values, marker='o', color='cyan')

    ax.set_xlabel("Month", fontsize=12, color='white')
    ax.set_ylabel("Total Amount Collected ($)", fontsize=12, color='white')
    ax.set_title("Monthly Waste Collection Trend", fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xticks(rotation=45, color='white')  # Rotate x-axis labels for better visibility
    plt.yticks(color='white')  # Set y-ticks color
    ax.set_facecolor('#222222')  # Dark background for the plot area
    fig.patch.set_facecolor('#333333')  # Dark gray background

    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from datetime import datetime
from sklearn.linear_model import LinearRegression
def user_waste_analysis(account_number, email):
    st.header("Your Waste Generation Analysis")

    waste_data = list(grabage_items_collection.find({
        "account_number": account_number,
        "email": email
    }))

    if not waste_data:
        st.warning("No waste data found for your account.")
        return

    df = pd.DataFrame(waste_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Waste Category Breakdown
    st.subheader("Waste Category Breakdown")
    category_counts = df["item_type"].value_counts()

    # Create a bar plot with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax, palette="coolwarm")

    ax.set_xlabel("Waste Category", fontsize=12, color='white')
    ax.set_ylabel("Count", fontsize=12, color='white')
    ax.set_title("Waste Category Breakdown", fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xticks(rotation=45, color='white')  # Rotate x-axis labels for better visibility
    plt.yticks(color='white')  # Set y-ticks color
    fig.patch.set_facecolor('#333333')  # Dark gray background

    st.pyplot(fig)

    # Waste Collection Over Time
    st.subheader("Waste Collection Over Time")
    df["date"] = df["timestamp"].dt.date
    daily_waste = df.groupby("date")["amount"].sum()

    # Create a line plot with dark theme
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_waste.index, daily_waste.values, marker='o', color='cyan')

    ax.set_xlabel("Date", fontsize=12, color='white')
    ax.set_ylabel("Amount Collected ($)", fontsize=12, color='white')
    ax.set_title("Daily Waste Collection", fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xticks(rotation=45, color='white')  # Rotate x-axis labels for better visibility
    plt.yticks(color='white')  # Set y-ticks color
    ax.set_facecolor('#222222')  # Dark background for the plot area
    fig.patch.set_facecolor('#333333')  # Dark gray background

    st.pyplot(fig)

import matplotlib.dates as mdates


def predict_future_waste_trends(account_number, email):
    """Predict future waste trends based on collected data."""
    st.header("Future Waste Trend Prediction")

    waste_data = list(grabage_items_collection.find({
        "account_number": account_number,
        "email": email
    }))

    if not waste_data:
        st.warning("No data available for prediction.")
        return

    df = pd.DataFrame(waste_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Ensure 'date' column is a datetime object
    df["date"] = pd.to_datetime(df["date"])

    daily_waste = df.groupby("date")["amount"].sum().reset_index()

    # Convert date to numerical values for prediction
    daily_waste["date_numeric"] = (daily_waste["date"] - daily_waste["date"].min()).dt.days
    X = daily_waste[["date_numeric"]]
    y = daily_waste["amount"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array(range(daily_waste["date_numeric"].max() + 1, daily_waste["date_numeric"].max() + 8)).reshape(
        -1, 1)
    future_predictions = model.predict(future_days)

    future_dates = pd.date_range(start=daily_waste["date"].max(), periods=7, freq='D')

    # Create a dark-themed plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(daily_waste["date"], daily_waste["amount"], label="Past Waste Collection", marker='o', color='cyan')
    ax.plot(future_dates, future_predictions, label="Predicted Waste Collection", linestyle='dashed', marker='x',
            color='red')

    ax.set_xlabel("Date", fontsize=12, color='white')
    ax.set_ylabel("Amount Collected ($)", fontsize=12, color='white')
    ax.set_title("Predicted Waste Trends", fontsize=14, color='white')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Format x-axis for better readability
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Set major ticks to every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format the date
    plt.xticks(rotation=45, color='white')  # Rotate date labels for better visibility
    plt.yticks(color='white')  # Set y-ticks color

    # Set background color
    fig.patch.set_facecolor('#333333')  # Dark gray background

    st.pyplot(fig)