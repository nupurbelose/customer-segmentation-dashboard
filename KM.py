import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the clustered dataset
data = pd.read_csv('clustered_customer_data.csv')

# Title for the app
st.title("💼 Customer Segmentation and Financial Dashboard")

# --- Sections in Expanders ---
with st.expander("📊 Predict Your Customer Segment", expanded=True):
    # Input fields for new customer data
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Monthly Income (in Rupees)", min_value=20000, max_value=500000, value=20000)
        expenditure = st.number_input("Monthly Expenditure (in Rupees)", min_value=0, max_value=500000, value=0)
    with col2:
        savings = st.number_input("Monthly Savings (in Rupees)", min_value=0, max_value=500000, value=0)
        investments = st.number_input("Monthly Investments (in Rupees)", min_value=0, max_value=50000, value=0)
    
    debt_level = st.number_input("Debt Level (in Rupees)", min_value=0, max_value=100000, value=0)

    # Prepare the new data for prediction
    features = np.array([[income, expenditure, savings, investments, debt_level]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaler.transform(data[['Income', 'Expenditure', 'Savings', 'Investments', 'Debt_Level']]))

    # Predict the cluster for the new data
    predicted_cluster = kmeans.predict(X_scaled)

    # Display the predicted cluster
    st.markdown(f"### 🎯 Predicted Segment: **{predicted_cluster[0]}**")

# --- Financial Dashboard ---
with st.expander("💰 Financial Dashboard", expanded=True):
    st.write("Here is a breakdown of your financials over the year:")

    # Calculate yearly figures
    yearly_income = income * 12
    yearly_expenditure = expenditure * 12
    yearly_savings = savings * 12
    yearly_investments = investments * 12

    # Display yearly figures in columns for better readability
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Yearly Income", f"₹{yearly_income}")
        st.metric("Yearly Expenditure", f"₹{yearly_expenditure}")
    with col2:
        st.metric("Yearly Savings", f"₹{yearly_savings}")
        st.metric("Yearly Investments", f"₹{yearly_investments}")
        
    # Create a financial summary
financial_summary = {
    "Yearly Income": yearly_income,
    "Yearly Expenditure": yearly_expenditure,
    "Yearly Savings": yearly_savings,
    "Yearly Investments": yearly_investments
}

# Personalized Suggestions Function
def suggest_strategy(yearly_savings, yearly_investments, goal_amount, goal_years):
    goal_total = goal_years * 12
    if yearly_savings + yearly_investments >= goal_amount:
        return "🎉 You're on track to achieve your financial goal!"
    else:
        extra_needed = (goal_amount - (yearly_savings + yearly_investments)) / goal_total
        return f"🔔 You need to save or invest an additional **₹{extra_needed:.2f}** per month to reach your goal."

# Risk Assessment Function
def risk_assessment(age, income, current_savings, investment_experience):
    if age < 30 and income > 50000 and investment_experience == "High":
        return "High"
    elif 30 <= age < 50 and income > 40000 and investment_experience == "Medium":
        return "Medium"
    else:
        return "Low"

# --- Choose Your Financial Path ---
with st.expander("🛣️ Choose Your Financial Path", expanded=True):
    option = st.selectbox("Would you like to focus on Savings or Investments?", ["Savings", "Investments"])

    if option == "Savings":
        savings_option = st.selectbox("Select a savings product:", ["Fixed Deposit", "Recurring Deposit", "Savings Account"])
        amount = st.number_input("Enter amount to save monthly (in Rupees)", min_value=0, value=savings)

        if savings_option == "Fixed Deposit":
            interest_rate = 0.06  # Assume a fixed interest rate of 6%
        elif savings_option == "Recurring Deposit":
            interest_rate = 0.07  # Assume a fixed interest rate of 7%
        else:
            interest_rate = 0.04  # Assume a fixed interest rate of 4% for savings account

        monthly_rate = interest_rate / 12  # Monthly interest rate
        months = 12  # 12 months in a year
        future_value_savings = amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

        st.success(f"📈 Estimated value after 1 year: **₹{future_value_savings:.2f}**")

    elif option == "Investments":
        investment_option = st.selectbox("Select an investment type:", ["Mutual Funds", "Stocks", "Bonds"])
        invest_amount = st.number_input("Enter amount to invest monthly (in Rupees)", min_value=0, value=investments)

        if investment_option == "Mutual Funds":
            expected_return = 0.12  # Assume an expected return of 12% per annum
        elif investment_option == "Stocks":
            expected_return = 0.15  # Assume an expected return of 15% per annum
        else:
            expected_return = 0.08  # Assume an expected return of 8% per annum

        monthly_rate = expected_return / 12  # Monthly rate of return
        months = 12  # 12 months in a year
        future_value_investment = invest_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

        st.success(f"📊 Estimated value after 1 year: **₹{future_value_investment:.2f}**")

# --- Financial Goals ---
with st.expander("🎯 Set Your Financial Goals", expanded=True):
    goal_name = st.text_input("What is your financial goal? (e.g., Vacation, Retirement)")
    goal_amount = st.number_input("Enter the target amount (in Rupees)", min_value=0)
    goal_years = st.number_input("In how many years do you want to achieve this goal?", min_value=1)

    if goal_name and goal_amount > 0 and goal_years > 0:
        # Calculate monthly savings needed to reach the goal
        monthly_needed = goal_amount / (goal_years * 12)
        st.write(f"📝 You need to save **₹{monthly_needed:.2f}** per month to achieve your goal of **₹{goal_amount}** in **{goal_years} years**.")
        
        # Provide personalized suggestions
        suggestion = suggest_strategy(yearly_savings, yearly_investments, goal_amount, goal_years)
        st.info(suggestion)
    else:
        st.write("Please fill in all fields to calculate your monthly savings needed.")

# --- Risk Assessment ---
with st.expander("⚖️ Risk Assessment", expanded=True):
    age = st.number_input("Enter your age", min_value=18, max_value=100, value=30)
    experience = st.selectbox("Investment Experience", ["Low", "Medium", "High"])
    risk_profile = risk_assessment(age, yearly_income, yearly_savings, experience)

    st.write(f"💡 Your risk profile is: **{risk_profile}**")

    # Suggest different investment products based on risk profile
    if risk_profile == "High":
        st.write("You should consider **higher risk, high-return options** such as stocks or aggressive mutual funds.")
    elif risk_profile == "Medium":
        st.write("You should consider a **balanced portfolio** with a mix of stocks and bonds.")
    else:
        st.write("You should consider **low-risk options** like bonds or fixed deposits.")

# --- Financial Breakdown Visualization ---
with st.expander("📊 Financial Breakdown", expanded=True):
    labels = list(financial_summary.keys())
    sizes = list(financial_summary.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
