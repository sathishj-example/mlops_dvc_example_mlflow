import pandas as pd
import pickle

# Load model
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load new data
new_data = pd.read_csv("data/new_customers.csv")

# Prepare features
X_new = new_data[["Age", "Tenure", "Balance", "Products", "IsActiveMember", "EstimatedSalary"]]

# Predict in batch
predictions = model.predict(X_new)

# Add predictions to data
new_data["Predicted_Churn"] = predictions

# Save output
new_data.to_csv("data/batch_predictions.csv", index=False)

print("Batch inference complete! Results saved to data/batch_predictions.csv")
