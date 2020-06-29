from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score    # 사이킷런에 있는 평가 지표 함수들


# 선형회귀모델 불러오기
lr = LinearRegression()

boston=pd.read_csv("./boston_house.csv")
print(boston.head(5))

feature_data = boston.iloc[:,:-1]
feature_names = boston.columns[:-1]

target_data = boston.iloc[:,-1]
target_names = boston.columns[-1]

x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3, random_state=1)
print(x_train.shape, y_train.shape)

lr.fit(x_train, y_train)
y_preds=lr.predict(x_test)


mse = mean_squared_error(y_test, y_preds)

rmse = np.sqrt(mse)

print("MSE : {0:.3f}, RMSE : {1:.3F}".format(mse, rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))