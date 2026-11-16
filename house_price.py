import pandas as pd
import numpy as np
data = pd.read_csv("data/houses.csv")

print(data.head())
print(data.describe())
print(data.info())

x = data[["size","bedrooms","bathrooms"]]
y = data["price"]
print(x)
print(y)
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training data:")
print(x_train)
print("Testing data:")
print(x_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print("Model coefficients:", model.coef_)
print("model intercept:", model.intercept_)
new_house = pd.DataFrame([[125, 4, 3]], columns=["size", "bedrooms", "bathrooms"])
predicted_price = model.predict(new_house)
print("Predicted house price:", predicted_price[0])
from sklearn.metrics import mean_absolute_error
predictions = model.predict(x_test)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)