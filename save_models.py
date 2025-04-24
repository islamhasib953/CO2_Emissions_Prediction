import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/CO2_Emissions_Canada.csv")

# Select features and target
features = [
    "Make", "Model", "Vehicle Class", "Engine Size(L)",
    "Transmission", "Fuel Type", "Fuel Consumption Hwy (L/100 km)"
]
target = "CO2 Emissions(g/km)"

# Create df_new
df_new = data[features].copy()

# Encode categorical variables
label_encoders = {}
for column in ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]:
    le = LabelEncoder()
    df_new[column] = le.fit_transform(df_new[column])
    label_encoders[column] = le

# Save label encoders for GUI
joblib.dump(label_encoders, "models/label_encoders.pkl")

# Prepare features (X) and target (y)
X = df_new.values  # Convert to NumPy array directly (no feature names)
y = data[target].values

# Scale numerical features (columns 3 and 6: Engine Size and Fuel Consumption)
scaler = StandardScaler()
X[:, [3, 6]] = scaler.fit_transform(X[:, [3, 6]])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and scaler
joblib.dump(model, "models/linear_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model, scaler, and label encoders saved successfully!")
