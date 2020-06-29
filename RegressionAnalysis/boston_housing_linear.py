import os
import pandas as pd
import numpy as np
import statsmodels.api as sm



# read data set
boston = pd.read_csv("./Boston_house.csv")

#check data set in a variety of way
print("print out head of data set\n", boston.head(5))
print("print out tail of data set\n", boston.tail(5))
print("check data size\n", boston.shape)
#check data info
boston.info()
#check basic info
boston.describe()

# STEP1. analysis of simple regression model
print("check data columns: ", boston.columns)

# feature data of boston
crim = boston[['CRIM']]
# target data
target = boston[['Target']]

crim_1 = sm.add_constant(crim, has_constant='add')
crim_1.head()
single_model = sm.OLS(target, crim_1)
fitted_model = single_model.fit()

# STEP2. print out and translate the result
print(fitted_model.summary())

# STEP3. prediction
crim_predict = fitted_model.predict(crim_1)
print(crim_predict)

import matplotlib.pyplot as plt

plt.scatter(crim, target, label="data")
plt.plot(crim, crim_predict, label="result")
plt.legend()
plt.show()

fitted_model.resid.plot()
plt.xlabel("residual")
plt.show()


lstat = boston[['LSTAT']]
lstat_l = sm.add_constant(lstat, has_constant='add')
lstat_model = sm.OLS(target, lstat_l)
lstat_fitted_model = lstat_model.fit()
print(lstat_fitted_model.summary())

lstat_predict = lstat_fitted_model.predict(lstat_l)
print(lstat_predict)

plt.scatter(lstat, target, label="data")
plt.plot(lstat, lstat_predict, label="predict")
plt.legend()
plt.show()

lstat_fitted_model.resid.plot()
plt.xlabel("residual_number")
plt.show()