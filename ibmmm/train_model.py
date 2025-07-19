import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Select features and target
selected_features = ['Age', 'Education', 'JobLevel', 'DistanceFromHome', 'TotalWorkingYears', 'JobRole', 'Department']
target = 'MonthlyIncome'

# Encode categorical features
for col in ['JobRole', 'Department']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Extract X and y
X = df[selected_features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open("salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'salary_model.pkl'")
