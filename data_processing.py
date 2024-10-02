import numpy as np
import pandas as pd

data = pd.read_excel('/content/data_raw.xlsx')
data.head()

summary=data.describe()
summary

missing_data=data.isnull().sum()
missing_data[missing_data > 0]

# encoding

from sklearn.preprocessing import LabelEncoder

one_hot_encoder_instance = LabelEncoder()

for column in data.select_dtypes(include=['object']).columns:
  data[column]=one_hot_encoder_instance.fit_transform(data[column])

print(data)

filtered_data=data.select_dtypes(include=[np.number])
filtered_data=filtered_data.dropna() # drop the missing value

missing_data_new=filtered_data.isnull().sum()
missing_data_new[missing_data > 0]

print("before normalization")
filtered_data.head()

from sklearn.preprocessing import StandardScaler

# from the 2nd to the 4th to the last
input_columns=filtered_data.columns[1:-3]

scaler=StandardScaler() # create an instance from a class
scaled_data=scaler.fit_transform(filtered_data[input_columns]) # call a method of the instance
filtered_data[input_columns]=scaled_data
scaled_data=filtered_data
scaled_data.head()

# for testing the index of the feature
print(filtered_data.columns[148])
print(filtered_data.columns[-3])
print(filtered_data.columns[148:-3])

print(filtered_data.columns[1:-3:50]) # for a single feature

filtered_data.iloc[:,[100,120,148]] # select multiple features

# select the input data (from the 2nd to the 4th to last columns)
scaled_data=np.asarray(scaled_data)
# input_data= scaled_data[:, 1:-3] # select all features
input_data= scaled_data[:, 148:-3] # select the features for S4
# input_data= scaled_data[:, 1:-3:50] # select features with equal spacing (useful for a single feature across seasons)
# input_data= scaled_data[:,[100,120,148]] # select multiple features
print(input_data)

# select the output data (the 3rd to last column)

# output_data = scaled_data [:,-3] # select data for output data "-3"
output_data = scaled_data [:,-2] # for adjusted salaries
print(output_data)

# split into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.33, random_state=0)

# import xgboost as xgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Best parameters from previous tuning
# best_params = {
#     'learning_rate': 0.1,
#     'n_estimators': 1000,
#     'max_depth': 7,
#     'min_child_weight': 1,
#     'gamma': 0.3,
#     'subsample': 0.7,
#     'colsample_bytree': 0.8,
#     'reg_alpha': 10,
#     'reg_lambda': 10,
#     'scale_pos_weight': 5,
#     'device': 'cuda'  # Using GPU
# }

# # Initialize the model with the best parameters
# model = xgb.XGBRegressor(**best_params)

# # Define KFold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Lists to store results
# rmse_list = []

# # Cross-validation loop
# for train_index, test_index in kf.split(input_data):
#     # Split the data into train and test sets
#     X_train, X_test = input_data[train_index], input_data[test_index]
#     y_train, y_test = output_data[train_index], output_data[test_index]

#     # Fit the model on the training data
#     model.fit(X_train, y_train)

#     # Predict on the test data
#     predicted_y_test = model.predict(X_test)

#     # Calculate RMSE for the current fold
#     rmse = np.sqrt(mean_squared_error(y_test, predicted_y_test))
#     rmse_list.append(rmse)

#     print(f"Fold RMSE: {rmse}")

# # Calculate average RMSE across all folds
# average_rmse = np.mean(rmse_list)

# # Print the cross-validation result
# print(f"Average RMSE across all folds: {average_rmse}")

print(y_test)