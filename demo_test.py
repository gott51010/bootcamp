# Databricks notebook source
! pip install --upgrade pip
! pip install kaggle
! pip install scikit-learn
! pip install pandas

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC cd /Workspace/Users/51010gotoh@gmail.com/taxi/
# MAGIC mkdir -p .kaggle
# MAGIC cd .kaggle
# MAGIC cp /Workspace/Users/51010gotoh@gmail.com/taxi/kaggle_20231106.json ./kaggle.json
# MAGIC chmod 600 kaggle.json
# MAGIC pwd
# MAGIC ls -la 

# COMMAND ----------

import os
import shutil
import pandas as pd
import numpy as np
import sklearn

dbutils.fs.put("/Workspace/Users/51010gotoh@gmail.com/taxi/train.csv", "path_to_local_train.csv", True)

# COMMAND ----------

df_train = pd.read_csv("/Workspace/Users/51010gotoh@gmail.com/taxi/train.csv")

# COMMAND ----------

df_train.head(10)

# COMMAND ----------

df_train.columns

# COMMAND ----------

df_train.info()

# COMMAND ----------

df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])

df_train['hour'] = df_train['pickup_datetime'].dt.hour
df_train['day_of_week'] = df_train['pickup_datetime'].dt.dayofweek

# COMMAND ----------

df_train.info()

# COMMAND ----------

df_train_class = pd.get_dummies(df_train, columns=['vendor_id', 'store_and_fwd_flag'])

# COMMAND ----------

df_train_class.info()

# COMMAND ----------

df_train_class['pickup_hour'] = pd.to_datetime(df_train_class['pickup_datetime']).dt.hour
df_train_class['pickup_day_of_week'] = pd.to_datetime(df_train_class['pickup_datetime']).dt.dayofweek
df_train_class['pickup_month'] = pd.to_datetime(df_train_class['pickup_datetime']).dt.month

df_train_class['dropoff_hour'] = pd.to_datetime(df_train_class['dropoff_datetime']).dt.hour
df_train_class['dropoff_day_of_week'] = pd.to_datetime(df_train_class['dropoff_datetime']).dt.dayofweek
df_train_class['dropoff_month'] = pd.to_datetime(df_train_class['dropoff_datetime']).dt.month

df_train_class = df_train_class.drop(columns=['pickup_datetime', 'dropoff_datetime'])


# COMMAND ----------

df_train_class.info()

# COMMAND ----------

df_train_class = df_train_class.drop(columns='id')

# COMMAND ----------

X = df_train_class.drop('trip_duration', axis=1)
y = df_train_class['trip_duration']

# COMMAND ----------

random_state = np.random.randint(0, 99999)
print("The random_state for this time = ", random_state) #  The random_state for this time = 48747
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# COMMAND ----------

from xgboost import XGBRegressor
# 実行時間２時間以上
# model = XGBRegressor()

# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     'max_depth': [2, 4, 8, 10],
#     'learning_rate': [0.01,0.05, 0.1, 0.2,0.3],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [0, 0.1, 0.5]
# }

# grid_search = sklearn.model_selection.GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error', 
#     cv=3,  
#     n_jobs=-1,  
#     verbose=2 
# )
# grid_search.fit(X_train, y_train)

# print("grid_search:", grid_search.best_params)

# model = XGBRegressor(n_estimators=100, learning_rate=0.3 ,max_bin=128, max_depth=8) #  Score: 0.580
# model.fit(X_train, y_train)


# COMMAND ----------

model = XGBRegressor(n_estimators=300, learning_rate=0.3, colsample_bytree=1, max_depth=7) #  Score:  0.66
# model = XGBRegressor()
model.fit(X_train, y_train)

# COMMAND ----------

y_pred = model.predict(X_test)

mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"MSE: {mse}")

# COMMAND ----------

rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# COMMAND ----------

mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# COMMAND ----------

r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(f" Score: {r2}")

# COMMAND ----------

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Trip Duration")
plt.ylabel("Predicted Trip Duration")
plt.title("Actual vs Predicted Trip Duration")
plt.show()