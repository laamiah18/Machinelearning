import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("fruits_data.csv")

# Convert date → month
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
df['month'] = df['date'].dt.month

# Replace wrong values
df = df.replace({"": None, " ": None, -99999: None, -88888: None})

# Handle missing values
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Interpolate each fruit column
fruit_cols = ['apple(1kg)', 'banana(1 dozen)', 'grapes(1kg)', 'mango(1kg)', 'Water Melons(1)']
for col in fruit_cols:
    df[col] = df[col].interpolate()

df = df.dropna()

# Train a model for each fruit
models = {}

for fruit in fruit_cols:
    X = df[['month']]
    y = df[fruit]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    filename = f"model_{fruit.split('(')[0]}.pkl"
    pickle.dump(model, open(filename, "wb"))
    
    print(f"Saved: {filename}")

print("All fruit models trained!")






















































