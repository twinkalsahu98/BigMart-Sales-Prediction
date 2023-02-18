# BigMart-Sales-Prediction
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a pandas DataFrame
df = pd.read_csv('sales_data.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1), df['Item_Outlet_Sales'], test_size=0.3, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance on the test data
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the model's performance metrics
print('Mean Squared Error: ', mse)
print('Root Mean Squared Error: ', rmse)
print('R-squared: ', r2)
