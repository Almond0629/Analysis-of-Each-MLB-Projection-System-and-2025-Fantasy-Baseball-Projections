from pybaseball import *
import pandas as pd
import math
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
import pickle

'''
      batting_stats index:
      Season, Name, Team: 1~3
      G, AB, PA, H: 5~8, 2B: 10, HR: 12, AVG: 24, BB%: 35, K%: 36
      OBP, SLG, OPS, ISO, BABIP: 38~42
      LD%, GB%, FB%: 44~46
      
      Others:
      a = stats[0:15] is a Pandas Dataframe
      to_numpy() converts into a 2D Numpy array
'''
    
def polyRegression(df, stat_name, rand_state):
    X = df[[stat_name,'Z-Swing%']]
    y = df[f'real_{stat_name}']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(x_train)
    X_poly_test = poly.fit_transform(x_test)
    model_train = Lasso(alpha=0.01)
    # model_train = LinearRegression()
    model_train.fit(X_poly_train, y_train)

    y_pred_train = model_train.predict(X_poly_train)
    y_pred_test = model_train.predict(X_poly_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # x_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    # x_range_poly = poly.fit_transform(x_range)
    # y_range_poly = model_train.predict(x_range_poly)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5, label='Test Data')
    # plt.plot(x_range, y_range_poly, color='red', linewidth=2, label='Regression Line')
    # plt.xlabel(f'Actual real_{stat_name}')
    # plt.ylabel(f'Predicted real_{stat_name}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return train_rmse, test_rmse, train_r2, test_r2

def xgboostRegressor(df, stat_name, rand_state):
    X = df[[stat_name,'CSW%']]
    y = df[f'real_{stat_name}'] 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    model = XGBRegressor(n_estimators=90, max_depth=2, learning_rate=0.03)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_rmse, test_rmse, train_r2, test_r2

# Statcast stat names: 'avg_hit_angle', 'fbld', 'ev95percent', 'brl_percent', 'Optimal_Pulled_Flyball%', '95%_EV'
def elasticNet(df, stat_name, rand_state):
    X = df[[stat_name,'Swing%']]
    y = df[f'real_{stat_name}'] 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return train_rmse, test_rmse, train_r2, test_r2
    
    # with open(f'./Projection_model/{stat_name}_model.pkl', 'wb') as file:
    #     pickle.dump(model, file)

projections = ['ATC','THE_BAT','THE_BAT_X','Steamer','ZIPS','ZIPS_DC','DC']
for proj in projections:
    path_1 = f'./Projection_model/Hitters/{proj}/2024_{proj}_Projections.csv'
    path_2 = f'./Projection_model/Hitters/{proj}/2023_{proj}_Projections.csv'
    path_1_df = pd.read_csv(path_1)
    path_2_df = pd.read_csv(path_2)
    path_df = pd.concat([path_1_df, path_2_df])
    path_1_df['HIP'] = path_1_df['H'] - path_1_df['HR']
    path_2_df['HIP'] = path_2_df['H'] - path_2_df['HR']
    path_df['HIP'] = path_df['H'] - path_df['HR']
    path_1_df['real_HIP'] = path_1_df['real_H'] - path_1_df['real_HR']
    path_2_df['real_HIP'] = path_2_df['real_H'] - path_2_df['real_HR']
    path_df['real_HIP'] = path_df['real_H'] - path_df['real_HR']
    path_1_df['1B%'] = path_1_df['1B'] / path_1_df['HIP']
    path_2_df['1B%'] = path_2_df['1B'] / path_2_df['HIP']
    path_df['1B%'] = path_df['1B'] / path_df['HIP']
    path_1_df['real_1B%'] = path_1_df['real_1B'] / path_1_df['real_HIP']
    path_2_df['real_1B%'] = path_2_df['real_1B'] / path_2_df['real_HIP']
    path_df['real_1B%'] = path_df['real_1B'] / path_df['real_HIP']
    path_1_df['2B%'] = path_1_df['2B'] / path_1_df['HIP']
    path_2_df['2B%'] = path_2_df['2B'] / path_2_df['HIP']
    path_df['2B%'] = path_df['2B'] / path_df['HIP']
    path_1_df['real_2B%'] = path_1_df['real_2B'] / path_1_df['real_HIP']
    path_2_df['real_2B%'] = path_2_df['real_2B'] / path_2_df['real_HIP']
    path_df['real_2B%'] = path_df['real_2B'] / path_df['real_HIP']
    path_1_df['3B%'] = path_1_df['3B'] / path_1_df['HIP']
    path_2_df['3B%'] = path_2_df['3B'] / path_2_df['HIP']
    path_df['3B%'] = path_df['3B'] / path_df['HIP']
    path_1_df['real_3B%'] = path_1_df['real_3B'] / path_1_df['real_HIP']
    path_2_df['real_3B%'] = path_2_df['real_3B'] / path_2_df['real_HIP']
    path_df['real_3B%'] = path_df['real_3B'] / path_df['real_HIP']
    path_1_df.to_csv(path_1, index=False)
    path_2_df.to_csv(path_2, index=False)
    stats = ['1B%','2B%','3B%']

    print(proj)
    for stat_name in stats:
        print(stat_name)
        print(np.sqrt(mean_squared_error(path_df[[f'{stat_name}']], path_df[f'real_{stat_name}'])))
        print(np.sqrt(mean_squared_error(path_1_df[[f'{stat_name}']], path_1_df[f'real_{stat_name}'])))
        print(np.sqrt(mean_squared_error(path_2_df[[f'{stat_name}']], path_2_df[f'real_{stat_name}'])))
        print()


# rand_state = [1,11,21,31,41,51]
# for num in rand_state:
#     # train_rmse, test_rmse, train_r2, test_r2 = polyRegression(path_df, stat_name, num)
#     train_rmse, test_rmse, train_r2, test_r2 = xgboostRegressor(path_df, stat_name, num)
#     # train_rmse, test_rmse, train_r2, test_r2 = elasticNet(path_df, stat_name, num)
#     print(f'Random state: {num}')
#     print(test_rmse)

# print(find_player_name('guerrero jr.','vladimir'))
# é