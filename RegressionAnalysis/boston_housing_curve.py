import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

boston = pd.read_csv("./boston_house.csv")
print(boston.head(5))

features = boston[['CRIM', 'RM', 'LSTAT']]
target = boston[['Target']]
print(features.head(3))

multi_features = sm.add_constant(features, has_constant='add')
multi_model = sm.OLS(target, multi_features)
fitted_multi_model = multi_model.fit()

print(fitted_multi_model.summary())

multi_pred = fitted_multi_model.predict(multi_features)
print(multi_pred)


import matplotlib.pyplot as plt

fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()

std_resid = fitted_multi_model.resid_pearson
plt.scatter(range(len(std_resid)), std_resid)
plt.show()