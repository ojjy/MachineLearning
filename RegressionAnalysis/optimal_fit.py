# Numpy와 pandas
import numpy as np
import pandas as pd

# 모델을 제작하고 평가하기 위한 사이킷런의 여러 패키지
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# 모델을 시각화하기 위한 라이브러리
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# 기본 설정
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.figsize'] = [9, 7]


# 랜덤값에 대한 시드 설정
np.random.seed(42)

# 사인 그래프로 데이터 생성
def true_gen(x):
    y = np.sin(1.2 * x * np.pi)
    return(y)

# 약간의 랜덤 노이즈 데이터가 더해진 x, y 값
x = np.sort(np.random.rand(120))
y = true_gen(x) + 0.1 * np.random.randn(len(x))


# Train data과 Test data을 만들기 위한 무작위 지수
random_ind = np.random.choice(list(range(120)), size = 120, replace=False)
xt = x[random_ind]
yt = y[random_ind]

# Train data와 Test data 분리
train = xt[:int(0.7 * len(x))]
test = xt[int(0.7 * len(x)):]

y_train = yt[:int(0.7 * len(y))]
y_test = yt[int(0.7 * len(y)):]


# 실제 곡선 모델링
x_linspace = np.linspace(0, 1, 1000)
y_true = true_gen(x_linspace)



# 관측값 시각화와 실제 곡선
plt.plot(train, y_train, 'ko', label = 'Train');
plt.plot(test, y_test, 'ro', label = 'Test')
plt.plot(x_linspace, y_true, 'b-', linewidth = 2, label = 'True Function')
plt.legend()
plt.xlabel('x'); plt.ylabel('y'); plt.title('Data');
plt.show()


def fit_poly(train, y_train, test, y_test, degrees, plot='train', return_scores=False):
    # 독립 변수들 다항식화하기 예시. 10개의 피쳐 10차원화 하기
    features = PolynomialFeatures(degree=degrees, include_bias=False)

    # 학습 데이터 재구성
    train = train.reshape((-1, 1))
    train_trans = features.fit_transform(train)

    # linear regression 모델 학습
    model = LinearRegression()
    model.fit(train_trans, y_train)

    # cross validation score 측정
    cross_valid = cross_val_score(model, train_trans, y_train, scoring='neg_mean_squared_error', cv=5)

    # 학습에 대한 예측과 오차 확인하기
    train_predictions = model.predict(train_trans)
    training_error = mean_squared_error(y_train, train_predictions)

    # 테스트 데이터 재구성
    test = test.reshape((-1, 1))
    test_trans = features.fit_transform(test)

    # 테스트셋에 대한 예측과 오차 확인하기
    test_predictions = model.predict(test_trans)
    testing_error = mean_squared_error(y_test, test_predictions)

    # 모델 곡선과 실제 곡선 찾기
    x_curve = np.linspace(0, 1, 100)
    x_curve = x_curve.reshape((-1, 1))
    x_curve_trans = features.fit_transform(x_curve)

    # 모델 곡선
    model_curve = model.predict(x_curve_trans)

    # 실제 곡선
    y_true_curve = true_gen(x_curve[:, 0])

    # 그래프 그리기 : 관측치, 실제 함수, 예측 함수 모형
    if plot == 'train':
        plt.plot(train[:, 0], y_train, 'ko', label='Observations')
        plt.plot(x_curve[:, 0], y_true_curve, linewidth=4, label='True Function')
        plt.plot(x_curve[:, 0], model_curve, linewidth=4, label='Model Function')
        plt.xlabel('x');
        plt.ylabel('y')
        plt.legend()
        plt.ylim(-1, 1.5);
        plt.xlim(0, 1)
        plt.title('{} Degree Model on Training Data'.format(degrees))
        plt.show()

    elif plot == 'test':
        # 그래프 그리기 : 관측치, 테스트셋 예측
        plt.plot(test, y_test, 'o', label='Test Observations')
        plt.plot(x_curve[:, 0], y_true_curve, 'b-', linewidth=2, label='True Function')
        plt.plot(test, test_predictions, 'ro', label='Test Predictions')
        plt.ylim(-1, 1.5);
        plt.xlim(0, 1)
        plt.legend(), plt.xlabel('x'), plt.ylabel('y');
        plt.title('{} Degree Model on Testing Data'.format(degrees)), plt.show();

    # 행렬 반환
    if return_scores:
        return training_error, testing_error, -np.mean(cross_valid)


fit_poly(train, y_train, test, y_test, degrees=1, plot='train')


fit_poly(train, y_train, test, y_test, degrees = 1, plot='test')

fit_poly(train, y_train, test, y_test, plot='train', degrees = 25)


fit_poly(train, y_train, test, y_test, degrees=25, plot='test')


fit_poly(train, y_train, test, y_test, plot='train', degrees = 5)

fit_poly(train, y_train, test, y_test, degrees=5, plot='test')



# cross validation
# 평가해볼 차수의 범위 설정(1차 ~ 40차)
degrees = [int(x) for x in np.linspace(1, 40, 40)]

# 결과에 대한 데이터 프레임 만들어주기
results = pd.DataFrame(0, columns = ['train_error', 'test_error', 'cross_valid'], index = degrees)

# N차 함수 모델 별로 오차에 대한 결과값을 생성해주기
for degree in degrees:
    degree_results = fit_poly(train, y_train, test, y_test, degree, plot=False, return_scores=True)
    results.loc[degree, 'train_error'] = degree_results[0]
    results.loc[degree, 'test_error'] = degree_results[1]
    results.loc[degree, 'cross_valid'] = degree_results[2]



print('10 Lowest Cross Validation Errors\n')
train_eval = results.sort_values('cross_valid').reset_index(level=0).rename(columns={'index': 'degrees'})
print(train_eval.loc[:,['degrees', 'cross_valid']].head(10))



plt.plot(results.index, results['cross_valid'], 'go-', ms=6)
plt.xlabel('Degrees'); plt.ylabel('Cross Validation Error'); plt.title('Cross Validation Results');
plt.ylim(0, 0.2);
print('Minimum Cross Validation Error occurs at {} degrees.\n'.format(int(np.argmin(results['cross_valid']))))



fit_poly(train, y_train, test, y_test, degrees=4, plot='train')

fit_poly(train, y_train, test, y_test, degrees=4, plot='test')


# 마지막으로 차수에 따른 MSE 시각화하기
plt.plot(results.index, results['train_error'], 'b-o', ms=6, label = 'Training Error')
plt.plot(results.index, results['test_error'], 'r-*', ms=6, label = 'Testing Error')
plt.legend(loc=2); plt.xlabel('Degrees'); plt.ylabel('Mean Squared Error'); plt.title('Training and Testing Curves');
plt.ylim(0, 0.05); plt.show()

print('\nTraining Error는 {}차 함수에서 최소값을 갖지만,'.format(int(np.argmin(results['train_error']))))
print('Testing Error는 {}차 함수에서 최소값을 갖는다.\n'.format(int(np.argmin(results['test_error']))))