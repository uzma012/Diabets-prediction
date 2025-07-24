import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
