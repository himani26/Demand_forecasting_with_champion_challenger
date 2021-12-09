import pandas as pd
import numpy as np
from dependencies import random_forest, runXGB, arima_model, prophet_model, ets_model, croston_model

def training_function(train_data,test_data):


    forecast = pd.DataFrame()
    forecast_output = pd.DataFrame()
    train_size = 160
    time = 4*6-1
    
    selected_skus1 = train_data['SKU'].unique()
    selected_skus = np.random.choice(selected_skus1, 1)
        
    for sku in selected_skus:

    #   Multivariate Modelling
        sub = train_data[train_data['SKU'] == sku].copy()
        sub.reset_index(drop=True, inplace =  True)

        future = test_data[test_data['SKU'] == sku].copy()
        TEST = future.copy()

        TEST.reset_index(drop=True, inplace =  True)

        sub_m = sub[['Date','PPI', 'CPI', 'Unemployment_Rate', 'Weekly Sales']]
        sub_m['Date'] = pd.to_datetime(sub_m['Date'])
        y = sub_m['Weekly Sales']
        X = sub_m.drop(['Weekly Sales'], axis = 1)
        X_train,X_test,y_train,y_test=X[:train_size],X[train_size:],y[:train_size],y[train_size:]

        x1 = X_test.index
        x2 = TEST.index

        X_train.set_index('Date', inplace=True)
        X_test.set_index('Date', inplace=True)
        TEST.set_index('Date', inplace=True)
        X.set_index('Date', inplace=True)

        RandomForest_pred, RandomForest_error, RandomForest_forecast = random_forest(X,y,X_train,X_test,y_train,y_test,TEST)
        xgb_pred, xgb_error, xgb_forecast = runXGB(X,y,X_train,X_test,y_train,y_test,TEST)

    #   Univariate Modelling    
        sub_u = sub[['Date','Weekly Sales','Categories']]
        sub_u['Date'] = pd.to_datetime(sub_u['Date'])
        train = sub_u[:train_size]
        test = sub_u[train_size:]

        arima_predictions, arima_error,arima_forecast = arima_model(sub_u, train, test,time)
        prophet_predictions, prophet_error,prophet_forecast = prophet_model(sub_u, train, test,time)
        ets_predictions, ets_error,ets_forecast = ets_model(sub_u,train['Weekly Sales'],test,time)
        croston_predictions, croston_error,croston_forecast = croston_model(pd.DataFrame(sub_u['Weekly Sales']), time = 4*6)

        # Combining forecasted results in one dataframe as forecast_output dataframe
        x3 = prophet_forecast.index
        RandomForest_forecast = pd.DataFrame(RandomForest_forecast, columns = ['RandomForest_Forecast'])
        xgb_forecast = pd.DataFrame(xgb_forecast, columns = ['XGB_Forecast'])

        prophet_forecast = pd.DataFrame(prophet_forecast)
        prophet_forecast.rename(columns = {'yhat': 'Prophet_Forecast'}, inplace = True)
        arima_forecast = pd.DataFrame(arima_forecast)
        arima_forecast.rename(columns = {'Forecast': 'Arima_Forecast'}, inplace = True)
        ets_forecast = pd.DataFrame(ets_forecast)
        ets_forecast.rename(columns = {'Forecast': 'ETS_Forecast'}, inplace = True)

        RandomForest_forecast.set_index(x3,drop=True, inplace =  True)
        xgb_forecast.set_index(x3,drop=True, inplace =  True)
        croston_forecast.set_index(x3,drop=True, inplace =  True)

        forecast = future.loc[future.index[x2]]
        forecast.set_index(x3,drop=True, inplace =  True)

        forecast1 = forecast.copy()
        forecast1['Prophet_Forecast'] = prophet_forecast
        forecast1['Arima_Forecast']  = arima_forecast
        forecast1['ETS_Forecast'] = ets_forecast
        forecast1['RandomForest_Forecast'] = RandomForest_forecast
        forecast1['XGB_Forecast'] = xgb_forecast
        forecast1['Croston_Forecast'] = croston_forecast

        forecast1['RandomForest_Error'] = RandomForest_error
        forecast1['XGB_Error']  = xgb_error
        forecast1['Arima_Error'] = arima_error
        forecast1['Prophet_Error'] = prophet_error
        forecast1['ETS_Error'] = ets_error
        forecast1['Croston_Error'] = croston_error

        forecast_output = forecast_output.append(forecast1)
        print(sku)
        
    cols1 = ['Date','Brands','Categories', 'Region', 'SKU', 
               'Prophet_Forecast','Prophet_Error','Arima_Forecast','Arima_Error', 'ETS_Forecast', 'ETS_Error',
             'RandomForest_Forecast','RandomForest_Error','XGB_Forecast','XGB_Error','Croston_Forecast','Croston_Error']

    forecast_output1 = forecast_output[cols1]
    
    return forecast_output1