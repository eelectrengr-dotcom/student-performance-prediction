import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# Features & target
X = df.drop("pass", axis=1)
y = df["pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LogisticRegression()
rf = RandomForestClassifier()

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Save models
pickle.dump(lr, open("logistic_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))

print("✅ Models trained and saved!")