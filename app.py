#First install streamlit using and category_encoders
#python -m pip install streamlit
#python -m pip install category_encoders

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import streamlit as st



dataset=pd.read_csv("Lab 21 House_Rent_Dataset.csv")
dataset.drop(columns=["Posted On","Point of Contact"],inplace=True)

x=dataset.drop(columns=["Rent"])
y=np.log1p(dataset["Rent"])

encoder=ce.LeaveOneOutEncoder(return_df=True)
x=encoder.fit_transform(x,y)


scaler=MinMaxScaler()
x=scaler.fit_transform(x)


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.20,random_state=40)

model_lr=LinearRegression()
model_lr.fit(x_train,y_train)
model_lr.score(x_train,y_train)
y_pred=np.expm1(model_lr.predict(x_test))
y_test=np.expm1(y_test)


mae=metrics.mean_absolute_error(y_test,y_pred)
mse=metrics.mean_squared_error(y_test,y_pred)
r2=metrics.r2_score(y_test,y_pred)

# ============================================================
#  PART 2: STREAMLIT CODE (the only new part)
# ============================================================

st.title("🏠 House Rent Price Predictor")
st.write("Enter the house details and click **Predict Rent**.")


# ---- Inputs (one widget for each column the model was trained on) ----
city = st.selectbox("City", sorted(dataset["City"].unique()))

# show only the localities of the chosen city
localities = sorted(dataset[dataset["City"] == city]["Area Locality"].unique())
locality = st.selectbox("Area Locality", localities)

area_type = st.selectbox("Area Type", sorted(dataset["Area Type"].unique()))
furnishing = st.selectbox("Furnishing Status", sorted(dataset["Furnishing Status"].unique()))
tenant = st.selectbox("Tenant Preferred", sorted(dataset["Tenant Preferred"].unique()))

bhk = st.number_input("BHK", min_value=1, max_value=6, value=2)
bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2)
size = st.number_input("Size (sq ft)", min_value=100, max_value=8000, value=900, step=50)

floor_no = st.selectbox("Floor", ["Ground"] + [str(i) for i in range(1, 31)], index=1)
total_floors = st.number_input("Total floors in the building", min_value=1, max_value=80, value=4)


# ---- Predict ----
if st.button("Predict Rent"):

    # 1. one-row table with the SAME columns, in the SAME order, as x
    new_house = pd.DataFrame([{
        "BHK": bhk,
        "Size": size,
        "Floor": f"{floor_no} out of {total_floors}",     # e.g. "1 out of 4"
        "Area Type": area_type,
        "Area Locality": locality,
        "City": city,
        "Furnishing Status": furnishing,
        "Tenant Preferred": tenant,
        "Bathroom": bathroom,
    }])



    # 2. apply the SAME steps used in training (transform, not fit_transform)
    new_house = encoder.transform(new_house)
    new_house = scaler.transform(new_house)

    # 3. predict (this is log of rent) and convert back to rupees
    log_rent = model_lr.predict(new_house)[0]
    rent = np.expm1(log_rent)

    st.success(f"Estimated monthly rent: ₹ {rent:,.0f}")

st.caption(f"Model accuracy on test data: R² = {r2:.2f}, MAE = ₹ {mae:,.0f}")








