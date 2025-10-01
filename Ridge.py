import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys


# Load the data
try:
    # Assuming 'melb_data.csv' is available
    data = pd.read_csv("melb_data.csv")
except FileNotFoundError:
    print("Error: 'melb_data.csv' not found. Please ensure it is in the same directory as the script.")
    sys.exit(1)


# Column to fill: Car
car_mode = data['Car'].mode()[0]
data['Car'] = data['Car'].fillna(car_mode)

# Column to fill: BuildingArea (Grouped by Rooms)
unique_rooms = data['Rooms'].unique()
for r in unique_rooms:
    rows_in_group = data['Rooms'] == r
    median_ba = data.loc[rows_in_group, 'BuildingArea'].median()
    if not np.isnan(median_ba):
        data.loc[rows_in_group, 'BuildingArea'] = data.loc[rows_in_group, 'BuildingArea'].fillna(median_ba)
# Global median fill for any remaining NaNs after grouping
global_median_ba = data['BuildingArea'].median(skipna=True)
data['BuildingArea'] = data['BuildingArea'].fillna(global_median_ba)


# Column to fill: YearBuilt (Grouped by Suburb)
unique_suburbs = data['Suburb'].unique()
for s in unique_suburbs:
    rows_in_suburb = data['Suburb'] == s
    median_yb = data.loc[rows_in_suburb, 'YearBuilt'].median()
    if not np.isnan(median_yb):
        data.loc[rows_in_suburb, 'YearBuilt'] = data.loc[rows_in_suburb, 'YearBuilt'].fillna(median_yb)

# Fill remaining missing YearBuilt values with the overall median
global_median_yb = data['YearBuilt'].median(skipna=True)
data['YearBuilt'] = data['YearBuilt'].fillna(global_median_yb)

# Column to fill: CouncilArea (Grouped by Suburb, then global mode)
for s in unique_suburbs:
    rows_in_suburb = data['Suburb'] == s
    known_council_areas = data.loc[rows_in_suburb, 'CouncilArea'].dropna()
    if not known_council_areas.empty:
        mode_council = known_council_areas.mode().iloc[0]
        data.loc[rows_in_suburb, 'CouncilArea'] = data.loc[rows_in_suburb, 'CouncilArea'].fillna(mode_council)

global_mode_ca = data['CouncilArea'].mode().iloc[0]
data['CouncilArea'] = data['CouncilArea'].fillna(global_mode_ca)


data = data.drop(columns=['Address', 'Date', 'SellerG'])



numeric_cols = ['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2',
               'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
               'Lattitude', 'Longtitude', 'Propertycount']

target_mu = data['Price'].mean()
target_sigma = data['Price'].std()

for colName in numeric_cols:
    col_data = data[colName]
    mu = col_data.mean()
    sigma = col_data.std()

    if sigma != 0:
        data[colName] = (col_data - mu) / sigma
    else:
        data[colName] = 0

# Final check to ensure NO NaNs remain in numeric columns after normalization
for colName in numeric_cols:
    if data[colName].isnull().any():
        global_median = data[colName].median(skipna=True)
        data[colName] = data[colName].fillna(global_median)


todo = ['Suburb', 'Type', 'Method', 'CouncilArea', 'Regionname']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(data[todo])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(todo))    

data = data.drop(columns=todo).reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)

# Combine numeric and encoded features
data = pd.concat([data, encoded_df], axis=1)

# Separate Features (X) and Target (y)
X = data.drop(columns=['Price']).values
y = data['Price'].values

print(f"Total features (columns): {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()


# 'alpha' is the regularization parameter, equivalent to 'lambda'
lambda_val = 0.1
alpha = lambda_val # Ridge uses L2 regularization, penalizing the sum of squared coefficients

print(f"\nTraining Ridge model with alpha (lambda) = {alpha}...")

# Initialize and train the Ridge model
# KEY CHANGE: Using Ridge instead of Lasso
ridge_model = Ridge(alpha=alpha, max_iter=3000, tol=1e-4)
ridge_model.fit(X_train, y_train_flat)



# Predict on test set
y_pred_ridge = ridge_model.predict(X_test)

# Calculate metrics
rmse_ridge = np.sqrt(mean_squared_error(y_test_flat, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test_flat, y_pred_ridge)

print("\n--- Scikit-learn RIDGE Results ---")
print(f"Alpha (lambda): {alpha}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"MAE: {mae_ridge:.4f}")
