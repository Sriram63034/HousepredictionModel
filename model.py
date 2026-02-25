"""
House Price Predictor for Vadodara Real Estate
Uses simple linear regression to predict house price based on size (sq ft)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("housePrice.csv")
X = data["Size_sqft"].values.reshape(-1, 1)  # sklearn expects 2D array
y = data["Price_lakhs"].values

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

print("=" * 50)
print("House Price Prediction Model - Vadodara")
print("=" * 50)
print(f"\nModel trained on {len(data)} data points")
print(f"Equation: Price (lakhs) = {model.coef_[0]:.4f} × Size (sqft) + {model.intercept_:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
print()

# Create the visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="steelblue", s=80, edgecolors="white", linewidth=1.5, label="Historical Data")

# Plot the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred_line = model.predict(X_line)
plt.plot(X_line, y_pred_line, color="coral", linewidth=2, label="Predicted Relationship")

plt.xlabel("House Size (sq ft)", fontsize=12)
plt.ylabel("Price (lakhs ₹)", fontsize=12)
plt.title("House Size vs Price - Vadodara Real Estate", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("size_vs_price.png", dpi=150, bbox_inches="tight")
print("Graph saved as 'size_vs_price.png'")
# Show graph (non-blocking so you can also use predictions)
plt.show(block=False)

# Interactive prediction
print("\n" + "-" * 50)
print("Price Prediction")
print("-" * 50)

while True:
    user_input = input("\nEnter house size (sq ft) to predict price (or 'q' to quit): ").strip()
    
    if user_input.lower() == "q":
        print("Thank you for using the House Price Predictor!")
        break
    
    try:
        size = float(user_input)
        if size <= 0:
            print("Please enter a positive number.")
            continue
            
        predicted_price = model.predict([[size]])[0]
        print(f"\n  Estimated Price: ₹{predicted_price:.2f} lakhs (~₹{predicted_price * 100000:,.0f})")
        
    except ValueError:
        print("Please enter a valid number.")
