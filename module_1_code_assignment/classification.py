# This is a beginner-friendly classification model.
# We will use Logistic Regression to predict whether
# road maintenance is urgent.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# random_state ensures that we get the same random results
# every time we run the program.
RANDOM_STATE = 42


# ============================================================
# STEP 1: CREATE A SYNTHETIC DATASET
# ============================================================

# Create a random-number generator.
rn_generator = np.random.default_rng(RANDOM_STATE)

# Number of fictional roads in the dataset.
number_roads = 1_000


# Number of potholes from 0 to 29.
potholes = rn_generator.integers(
    low=0,
    high=30,
    size=number_roads,
)

# Crack severity from 0 to 10.
crack_severity = rn_generator.uniform(
    low=0,
    high=10,
    size=number_roads,
)

# Estimated number of vehicles per day.
daily_traffic = rn_generator.integers(
    low=500,
    high=50_000,
    size=number_roads,
)

# Rainfall during a period, measured in millimeters.
rainfall_mm = rn_generator.uniform(
    low=0,
    high=300,
    size=number_roads,
)

# Age of the pavement, measured in years.
pavement_years = rn_generator.uniform(
    low=0,
    high=35,
    size=number_roads,
)


# ============================================================
# STEP 2: CREATE THE TARGET VARIABLE
# ============================================================

# This is the hidden relationship used to generate
# our synthetic target.
risk_score = (
    0.25 * potholes
    + 0.80 * crack_severity
    + 0.00004 * daily_traffic
    + 0.008 * rainfall_mm
    + 0.15 * pavement_years
    + rn_generator.normal(
        loc=0,
        scale=2,
        size=number_roads,
    )
)

# Use the 60th percentile as the urgency threshold.
# Approximately the highest 40% of scores will be urgent.
urgency_threshold_level = np.quantile(
    risk_score,
    0.60,
)

# The comparison creates Boolean values:
#
# False -> 0
# True  -> 1
urgent_maintenance = (
    risk_score >= urgency_threshold_level
).astype(int)


# ============================================================
# STEP 3: ORGANIZE THE DATA IN A PANDAS DATAFRAME
# ============================================================

data = pd.DataFrame(
    {
        "potholes": potholes,
        "crack_severity": crack_severity,
        "daily_traffic": daily_traffic,
        "rainfall_mm": rainfall_mm,
        "pavement_years": pavement_years,
        "urgent_maintenance": urgent_maintenance,
    }
)

print("Dataset Preview")
print("-" * 50)
print(data.head())

print("\nDataset Size")
print("-" * 50)
print(data.shape)

print("\nClass Distribution")
print("-" * 50)
print(data["urgent_maintenance"].value_counts())


# ============================================================
# STEP 4: SEPARATE FEATURES FROM THE TARGET
# ============================================================

feature_names = [
    "potholes",
    "crack_severity",
    "daily_traffic",
    "rainfall_mm",
    "pavement_years",
]

# X contains the model inputs.
X = data[feature_names]

# Y contains the correct classifications.
Y = data["urgent_maintenance"]

print("\nFeatures")
print("-" * 50)
print(X.head())

print("\nTarget")
print("-" * 50)
print(Y.head())


# ============================================================
# STEP 5: SPLIT THE DATA
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,

    # Reserve 20% of the dataset for testing.
    test_size=0.20,

    # Ensure reproducible results.
    random_state=RANDOM_STATE,

    # Preserve the class proportions in both sets.
    stratify=Y,
)

print("\nTraining and Testing Sets")
print("-" * 50)
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# ============================================================
# STEP 6: SCALE THE FEATURES
# ============================================================

# Scaling is a preprocessing operation.
# It transforms features with different ranges into
# comparable numerical scales.
scaler = StandardScaler()

# Learn the scaling values from the training data
# and transform the training data.
X_train_scaled = scaler.fit_transform(X_train)

# Only transform the testing data.
# Do not learn from the testing data.
X_test_scaled = scaler.transform(X_test)


# ============================================================
# STEP 7: CREATE THE CLASSIFICATION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1_000,
    random_state=RANDOM_STATE,
)


# ============================================================
# STEP 8: TRAIN THE MODEL
# ============================================================

model.fit(
    X_train_scaled,
    Y_train,
)

print("\nThe model has been trained!")


# ============================================================
# STEP 9: MAKE PREDICTIONS
# ============================================================

# Return final classes: 0 or 1.
predictions = model.predict(X_test_scaled)

# Return probabilities for each class.
# [:, 1] selects the probability of class 1.
urgent_probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

print("\nFirst Five Predictions")
print("-" * 50)

prediction_comparison = pd.DataFrame(
    {
        "actual_value": Y_test.iloc[:5].values,
        "predicted_value": predictions[:5],
        "urgent_probability": urgent_probabilities[:5],
    }
)

print(prediction_comparison)


# ============================================================
# STEP 10: EVALUATE THE MODEL
# ============================================================

accuracy = accuracy_score(
    Y_test,
    predictions,
)

print("\nModel Accuracy")
print("-" * 50)
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report")
print("-" * 50)

print(
    classification_report(
        Y_test,
        predictions,
        target_names=[
            "Not Urgent",
            "Urgent",
        ],
    )
)


# ============================================================
# STEP 11: DISPLAY THE CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    Y_test,
    predictions,
    display_labels=[
        "Not Urgent",
        "Urgent",
    ],
)

plt.title("Road Maintenance Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: CLASSIFY A NEW ROAD
# ============================================================

new_road = pd.DataFrame(
    [
        {
            "potholes": 18,
            "crack_severity": 7.8,
            "daily_traffic": 32_000,
            "rainfall_mm": 180,
            "pavement_years": 20,
        }
    ]
)

# Apply the same scaling learned from the training data.
new_road_scaled = scaler.transform(new_road)

new_prediction = model.predict(
    new_road_scaled
)[0]

new_probability = model.predict_proba(
    new_road_scaled
)[0, 1]

if new_prediction == 1:
    predicted_label = "Urgent"
else:
    predicted_label = "Not Urgent"

print("\nNew Road Classification")
print("-" * 50)
print(new_road.to_string(index=False))

print(f"\nPrediction: {predicted_label}")
print(
    f"Probability of urgent maintenance: "
    f"{new_probability:.2%}"
)