import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv("salary_train.csv")
test_df = pd.read_csv("salary_test.csv")

# Combine for consistent preprocessing
train_df["source"] = 1
test_df["source"] = 0
test_df["salary"] = np.nan

combined = pd.concat([train_df, test_df], ignore_index=True)

# Fill missing values
num_cols = combined.select_dtypes(include=["float64", "int64"]).columns
cat_cols = combined.select_dtypes(include=["object"]).columns

for col in num_cols:
    combined[col] = combined[col].fillna(combined[col].median())

label_encoders = {}
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0])
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le

# Split back
train = combined[combined["source"] == 1].drop(columns=["source"])
X = train.drop(columns=["ID", "salary"])
y = train["salary"]

# Ensemble model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
ensemble = VotingRegressor([("rf", rf), ("gb", gb)])
ensemble.fit(X, y)

# Save model and encoders
with open("salary_model.pkl", "wb") as f:
    pickle.dump((ensemble, label_encoders), f)

print("Model trained and saved to models/salary_model.pkl")
