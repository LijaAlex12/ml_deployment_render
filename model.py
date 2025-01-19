import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Sample data: area (in sq ft), rooms, and price (in $)
data = {
    'area': [1000, 1500, 1800, 2400, 3000],
    'rooms': [2, 3, 3, 4, 5],
    'price': [400000, 500000, 550000, 650000, 700000]
}

df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['area', 'rooms']]
y = df['price']

# Train-test split (although it's not needed for such small data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'house_price_model.pkl')

print("Model trained and saved successfully.")
