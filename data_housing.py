import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression,Ridge,Lasso,ElasticNet,SGDRegressor,HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import lightgbm as lgb
data = pd.read_csv(r"C:\Users\tanpu\Downloads\14th,- REGRESSION PROJECT\14th,- REGRESSION PROJECT\HOUSING REGRESSOR\USA_Housing.csv")
data.head()

X_ = data.drop(['Price', 'Address'], axis=1)
y_ = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=0)




models = {
    'LinearRegression': LinearRegression(),
    'RidgeRegression': Ridge(),
    'RobustRegression': HuberRegressor(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })

    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)

print("Model training and evaluation completed. Results saved to 'model_results.csv'.")



