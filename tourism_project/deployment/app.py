import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="parth1706/tourism-package-predictor", filename="tourism_v1_0.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase the newly introduced Wellness Tourism Package.
Please enter the customer details below and click Predict.
""")

# --------------------------------------------------
# Customer Inputs
# --------------------------------------------------

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

NumberOfTrips = st.number_input(
    "Number Of Trips Per Year",
    min_value=0,
    max_value=50,
    value=2
)

Passport = st.selectbox(
    "Passport Available",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    max_value=20,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=1,
    max_value=1000,
    value=15
)


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame([{

    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch

}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"Customer is likely to purchase the Wellness Tourism Package.\n\nProbability: {probability:.2%}"
        )

    else:

        st.error(
            f"Customer is unlikely to purchase the Wellness Tourism Package.\n\nProbability: {probability:.2%}"
        )

    st.subheader("Input Summary")
    st.dataframe(input_data)
