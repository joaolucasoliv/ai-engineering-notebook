# Now I will create a beginner-friendly Regression model
# Instead of predicting a category such as urgent and not urgent, we will predict A CONTINUOUS NUMBER
# We gonna ESTIMATED RODAD MAINTENANCE COST in dollars
# We will use number of potholes, crack severity, daily traffic, road length and age.
# The model learns how much each feature contributes to the final prediction. It finds the line—or, with multiple features, the mathematical plane—that minimizes the squared prediction errors.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from module_1_code_assignment.classification import RANDOM_STATE, number_roads, crack_severity, daily_traffic, \
    prediction_comparison

# random_state ensures that we generate the same random data -> every time we run the program

RANDOM_STATE = 42


# ======================
# step 1. CREATE A SYNTHETIC DATASET
# ======================

random_generator = np.random.default_rng(RANDOM_STATE) #check

#create the number of roads in our dataset (fictional)
number_roads = 1_000

# Number of potholes on each road - values range from 0 to 29
potholes = random_generator.integers(
        low=0,
        high=30,
        size=number_roads,
    )

# crack severity from 0 to 10 -> a higher value means more severe cracking.
crack_severity = random_generator.integers(
        low=500,
        high=25_000,
        size=number_roads,
    )

# estimated number of bechicles using the road every day.
daily_traffic  = random_generator.integers(
        low=500,
        high=25_000,
        size=number_roads,
    )

# length of the road in km.
road_length_km = random_generator.uniform(
        low=0.2,
        high=12,
        size=number_roads,
    )

# age of the road in years
road_age_years = random_generator.integers(
        low=1,
        high=40,
        size=number_roads,
    )


# ======================
# step 2. CREATE A TARGET VARIABLE -> MAINTENANCE COST
# ======================

# the target is the value we want the model to predict -> maintenance cost in dollars
# we create a fictional cost formula so that the model has realistic patterns to learn.

maintenance_cost = (
    5_000 + potholes * 850 + crack_severity * 4_000 + daily_traffic * 0.30 + road_length_km * 6_000 + road_age_years * 500
)

# real-world data is never perfectly predictable --> we can have noise
# noise represents unknown factors such as: material prices, labor availability, weather, regional differences, measurements errors...

noise = random_generator.normal( # "draws random samples from a Gaussian (normal) bell-curve distribution."
    loc=0,
    scale=8_000,
    size=number_roads,
)

maintenance_cost = maintenance_cost + noise

#prevent the synthetic cost from becoming negative!!!
maintenance_cost = np.maximum(maintenance_cost, 1_000)

# ======================
# step 3. CREATE A PANDAS DATAFRAME
# ======================

# A DataFrame will organize the information into rows and columns.

roads_data = pd.DataFrame(
    {
        "potholes": potholes,
        "crack_severity": crack_severity,
        "daily_traffic": daily_traffic,
        "road_length_km": road_length_km,
        "road_age_years": road_age_years,
        "maintenance_cost": maintenance_cost,
    }
)

#Display the first 10 roads.
print("\nFirst ten rows:")
print(roads_data.head())

# show basic information about the dataset
print("\nDataset Information:")
roads_data.info()

# show statistics such as mean, minimum, and maximum
print("\nDataset Stats:")
print(roads_data.describe())

# ======================
# step 4. SEPARATE FEATURES AND TARGET
# ======================

#features are the information used to make the prediction

feature = [
    "potholes",
    "crack_severity",
    "daily_traffic",
    "road_length_km",
    "road_age_years",
]
X = roads_data[feature]

y = roads_data["maintenance_cost"] # target value that we want to predict


# ======================
# step 5. SPLIT THE DATA
# ======================

# we divide the dataset into two parts: training data (learn relationships) and testing data (to evaluate the model on roads it has never seen)]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

print("\nTraining roads:", len(X_train))
print("\nTesting roads:", len(X_test))


# ======================
# step 6. CREATE AND TRAIN THE MODEL
# ======================

# create the LINEAR REGRESSION model
model = LinearRegression()

#fit() trains the model + the model studies thr training data and learns
# - the intercept
# - one coefficient for each feature

model.fit(X_train, y_train)

# ======================
# step 7. MAKE PREDICTIONS WITH THE DATA
# ======================

#predict() estimates maintenance costs for the testing roads.

y_predicted = model.predict(X_test)

#create a table comparing real and predicted values
prediction_comparison = pd.DataFrame(
    {
        "actual_cost": y_test,
        "predicted_cost": y_predicted,
    }
)

# reset the index to make the table easier to read
prediction_comparison = prediction_comparison.reset_index(drop=True)

print("\nPrediction comparison:")
print(prediction_comparison.head(10))


# ======================
# step 8. Evaluate the Model
# ======================

# mean absolute error (mae): the average absolute difference between the real cost and the predicted cost.

mae = mean_absolute_error(y_test, y_predicted)

# mean squared error (mse): the squares every error before calculating the average
# this gives more importance to large mistakess.

mse = mean_squared_error(y_test, y_predicted)


# Root Mean Squared Error: converts MSE back into the same unit as the target.

rmse = np.sqrt(mse)


# R-squared: measures how much of the variation in maintenance costs is explained by the model
# 1.0 means perfect conditions
# 0.0 means the model is ot better than predicting the average.
# negative means the model is worse than predicting the average.

r_squared = r2_score(y_test, y_predicted)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R-Squared Score: ${r_squared:.4f}") #check

# ======================
# step 9. INSPECT WHAT THE MODEL LEARNED
# ======================

# the intercept is the model's starting value before
# considering any of the road features.
print("\nModel intercept:")
print(f"${model.intercept_:,.2f}")

# each coefficient represents how much the prediction changes
# when that feature increases by one unit, while the other features remain unchanged

coefficient = pd.DataFrame(
    {
        "feature": feature,
        "coefficient": model.coef_,
    }
)

print("\nCoefficient:")
print(coefficient)

# print an easier-to-read explanation of each coefficient
print("\nCoefficient Interpretation:")

for feature, coefficient in zip(feature, model.coef_):
    print(
        f"A one-unit increase in {feature} changes the "
        f"predicted cost by approx. ${coefficient:,.2f}."
    )

# ======================
# step 10. VISUALIZE ACTUAL VS. PREDICTED COSTS
# ======================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_predicted,
    alpha=0.6, #check
)

# find the minimum and maximum values so that we can draw the perfect-prediction reference line

minimum_cost = min(y_test.min(), y_predicted.min())
maximum_cost = max(y_test.max(), y_predicted.max())

# Points directly on this line represents the perfect predictions
plt.plot (
    [minimum_cost, maximum_cost],
    [minimum_cost, maximum_cost], #check
    linestyle="--",
)

plt.xlabel("Actual Maintenance Cost")
plt.ylabel("Predicted Maintenance Cost")
plt.title("Actual Cost vs. Predicted Cost")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ======================
# step 11. VISUALIZE THE RESIDUALS
# ======================

# a residual is ACTUAL VALUE [y test] - PREDICTED VALUE [y predicted]
# Positive residual = the model predicted too little (predicted not enough)
# negative residual = the model predicted too much

residuals = y_test - y_predicted

plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    y_predicted,
    alpha=0.6,
)

# the horizonal line at zero represents A PERFECT PREDICTION.
plt.axhline(
    y=0,
    linestyle="--",
)

plt.xlabel("Predicted Maintenance Cost")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ======================
# step 12. PREDICT THE COST OF A NEW ROAD
# ======================

# create a new road using exactly the same feature columns that were used to train the model.

new_road = pd.DataFrame(
    {
        "potholes": [18],
        "crack_severity": [7.5],
        "daily_traffic": [15_000],
        "road_length_km": [4.2],
        "road_age_years": [15],
    }
)

# Ask the model to estimate the maintenance cost.
new_road_prediction = model.predict(new_road)

print("\nNew Road Prediction:")
print(new_road)

print(
    "\nEstimated maintenance cost: "
    f"${new_road_prediction[0]:,.2f}" #check
)

# ==============================
# ADDITIONAL STEP 12. INPUT TO EACH CLASS
# ==============================

print("\n ---Road Maintenance Cost Predictor---")
print("Enter the information below:\n")

# input always returns text -> so we can convert text answers into numbers using int() to whole numbers and flot() for decimals numbers

potholes_input = int(
    input("Number of potholes: ")
)

crack_severity_input = float(
    input("Crack Severity from 0 to 10:")
)

daily_traffic_input = int(
    input("Expected daily traffic per day:")
)

road_length_km_input = float(
    input("Road length in km: ")
)
road_age_years_input = int(
    input("Road age in years: ")
)

# We should create a DataFrame cointaining the road entered by the user -> columns exactly like the ones used to train the model.

new_road = pd.DataFrame(
    {
        "potholes": [potholes_input],
        "crack_severity": [crack_severity_input],
        "daily_traffic": [daily_traffic_input],
        "road_length_km": [road_length_km_input],
        "road_age_years": [road_age_years_input],
    }
)
# now create a function to train the model to predict the TARGET: MAINTENANCE COST

new_road_prediction = model.predict(new_road)
print(f"Number of potholes: {potholes_input}")
print(f"Crack severity: {crack_severity_input}")
print(f"Daily traffic: {daily_traffic_input:,} vehicles")
print(f"Road length: {road_length_km_input:.2f} km")
print(f"Road age: {road_age_years_input} years")

print(
    "\nEstimated maintenance cost: "
    f"${new_road_prediction[0]:,.2f}"
)