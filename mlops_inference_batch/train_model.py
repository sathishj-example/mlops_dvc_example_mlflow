import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = pd.read_csv("data/churn_data.csv")

# Features and target
X = data[["Age", "Tenure", "Balance", "Products", "IsActiveMember", "EstimatedSalary"]]
y = data["Churn"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
