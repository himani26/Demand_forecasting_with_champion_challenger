#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Load specific forecasting tools
from pmdarima import auto_arima   # for determining ARIMA orders
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
xgb.set_config(verbosity=0)
from croston import croston
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

def maape(target, forecast):
    ape = np.zeros_like(target, dtype="float")
    mask = np.logical_not((target == 0) & (forecast == 0))
    ape = np.divide(np.abs(target - forecast), target, out=ape, where=mask)
    return np.mean(np.arctan(ape))

def making(train_data,sku):

    train_data = train_data[train_data.Categories =='Dairy-based Drinks']
    new_product_skus = ['SKU_62','SKU_63','SKU_64','SKU_65']
    old_product_data = train_data[train_data.Date < '2013-02-17']
    new_product_data = train_data[train_data.SKU.isin(new_product_skus)]
    
    a1 = new_product_data[new_product_data.SKU ==sku][['Date', 'PPI', 'CPI', 'Unemployment_Rate','Temperature',
    'Discounts', 'Weekly Sales']].reset_index(drop = True)

    b1 = old_product_data.groupby('Date').mean().reset_index()

    abc = pd.concat([b1,a1], axis = 0).reset_index(drop = True)

    c1 = new_product_data[new_product_data.SKU ==sku][['Brands', 'Categories', 'Region', 'SKU']].reset_index(drop = True)

    df2 = pd.DataFrame()
    df2 = pd.concat([c1]*8, ignore_index=True)
    df2 = df2[:-10]

    final = pd.concat([abc,df2],axis = 1)
    cols = ['Date', 'Brands', 'Categories', 'Region', 'SKU','Temperature',
        'PPI', 'CPI', 'Unemployment_Rate', 'Discounts','Weekly Sales']
    final = final[cols]

    return final

def random_forest(X,y,X_train,X_test,y_train,y_test,TEST):
    
    model = RandomForestRegressor()

    # Choose some parameter combinations to try
    parameters = {'n_estimators': [5,10],
              'criterion': ['mse'],
              'max_depth': [5,10], 
              'min_samples_split': [2,5],
              'min_samples_leaf': [1,5]}


    #Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
    model_obj = GridSearchCV(model, parameters,
                            cv=5, 
                            n_jobs=-1, #Number of jobs to run in parallel
                            verbose=1)
    
    model_obj = model_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    model = model_obj.best_estimator_
    print(model_obj.best_estimator_)

    # Fit the best algorithm to the data. 
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    error = maape(y_test,pred)
    
    model.fit(X,y)
    forecast = model.predict(TEST)
    
    return pred, error, forecast

def runXGB(X,y,X_train,X_test,y_train,y_test, TEST):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.02 
    params["min_child_weight"] = 8
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 8
    params["seed"] = 1
    plst = list(params.items())
    num_rounds = 500

    xgtrain = xgb.DMatrix(X_train, label=y_train,enable_categorical = True)
    xgtest = xgb.DMatrix(X_test,enable_categorical = True)
    xgTEST = xgb.DMatrix(TEST,enable_categorical = True)
    xgX = xgb.DMatrix(X, label=y, enable_categorical = True)
    
    model = xgb.train(plst, xgtrain, num_rounds)
    pred = model.predict(xgtest)
    error = maape(y_test,pred)
    
    model = xgb.train(plst, xgX, num_rounds)
    forecast = model.predict(xgTEST)

    return pred, error, forecast
    
def arima_model(sub, train, test, time):
    train_size = 160
    model = auto_arima(sub['Weekly Sales'], seasonal=True, m=28)

    get_parametes = model.get_params()

    order_aa = get_parametes.get('order')
    seasonal_order_aa = get_parametes.get('seasonal_order')
    
    model_arima = SARIMAX(train['Weekly Sales'], 
                        order = (order_aa[0], order_aa[1], order_aa[2]),  
                        seasonal_order =(seasonal_order_aa[0], seasonal_order_aa[1], 
                                         seasonal_order_aa[2], seasonal_order_aa[3]), enforce_stationarity=False, 
                          enforce_invertibility=False) 

    result = model_arima.fit()

    start=len(train)
    end=len(train)+len(test)-1
    predictions = result.predict(start=start, end=end, dynamic=False, typ='levels').rename('Predictions')

    error = maape(test['Weekly Sales'], predictions)
    
    model_arima = SARIMAX(sub['Weekly Sales'], 
                        order = (order_aa[0], order_aa[1], order_aa[2]),  
                        seasonal_order =(seasonal_order_aa[0], seasonal_order_aa[1], 
                                         seasonal_order_aa[2], seasonal_order_aa[3]), enforce_stationarity=False, 
                          enforce_invertibility=False) 

    results = model_arima.fit()
    forecast = results.predict(len(sub),len(sub)+time,typ='levels').rename('Forecast')
        
    if (sub.Categories.unique() == 'Energy Drinks') | (sub.Categories.unique() == 'Beverage Mixes'):
        predictions = pd.DataFrame(index=range(len(sub)-train_size),columns=range(1)) 
        error = np.nan 
        forecast= pd.DataFrame(index=range(time+1),columns=['Forecast'])
    else:
        None
        
    return predictions,error,forecast

def prophet_model(sub, train, test,time):
    sub = sub[['Date','Weekly Sales']].rename({'Date':'ds','Weekly Sales':'y'}, axis = 1)    
    train = train[['Date','Weekly Sales']].rename({'Date':'ds','Weekly Sales':'y'}, axis = 1)    
    test = test[['Date','Weekly Sales']].rename({'Date':'ds','Weekly Sales':'y'}, axis = 1)
    
    model = Prophet()
    model.fit(train)
    preds=model.predict(test)

    predictions = preds['yhat']
    error = maape(test['y'].reset_index(drop = True),predictions)

    model = Prophet()
    model.fit(sub)
    future_dates=model.make_future_dataframe(periods=time+1,freq='W')
    fcs=model.predict(future_dates)
    forecast = fcs[len(sub):]['yhat']
    return predictions, error, forecast
    
def ets_model(sub,train,test,time):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=52, damped_trend=True)
        
    result = model.fit(optimized=True, use_boxcox=False, remove_bias=False)

    start=len(train)
    end=len(train)+len(test)-1
    predictions = result.predict(start=start, end=end).rename('Predictions')

    error = maape(test['Weekly Sales'], predictions)
    
    model = ExponentialSmoothing(sub['Weekly Sales'], trend='add', seasonal='add', seasonal_periods=52, damped_trend=True)
    result = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
    forecast = result.predict(len(sub),len(sub)+time).rename('Forecast')
    
    return predictions, error, forecast 

def croston_model(df, time = 4*6):
    train_size = 160
    fit_pred = croston.fit_croston(df,time,'original')
    pred = pd.DataFrame(fit_pred['croston_fittedvalues'][:train_size], columns =['Croston_Predictions'])
    error = maape(df[train_size:].reset_index(drop=True).to_numpy(),
                  pd.DataFrame(fit_pred['croston_fittedvalues'][train_size:]).to_numpy())
    forecast = pd.DataFrame(fit_pred['croston_forecast'], columns =['Croston_Forecast'])
    return pred, error,forecast

def new_product_data_preparation(train_data):
    x_62 = making(train_data,'SKU_62')
    x_63 = making(train_data,'SKU_63')
    x_64 = making(train_data,'SKU_64')
    x_65 = making(train_data,'SKU_65')

    X = pd.concat([x_62,x_63,x_64,x_65], axis = 0).reset_index(drop = True)
    new_product_skus = ['SKU_62','SKU_63','SKU_64','SKU_65']
    Y = train_data[~train_data.SKU.isin(new_product_skus)]

    master_data = pd.concat([X,Y], axis = 0).reset_index(drop = True)

    master_data = master_data.sort_values(['SKU','Date']).reset_index(drop = True)

    train_data = master_data.copy()
    return train_data

def training_function(train_data,test_data):

    forecast = pd.DataFrame()
    forecast_output = pd.DataFrame()
    train_size = 160
    time = 4*6-1
    
    selected_skus1 = train_data['SKU'].unique()
    selected_skus = np.random.choice(selected_skus1, 3)
        
    for sku in selected_skus:

    #   Multivariate Modelling
        sub = train_data[train_data['SKU'] == sku].copy()
        sub.reset_index(drop=True, inplace =  True)

        future = test_data[test_data['SKU'] == sku].copy()
        TEST = future.copy()

        TEST.reset_index(drop=True, inplace =  True)

        sub_m = sub[['Date','PPI', 'CPI', 'Unemployment_Rate', 'Weekly Sales']]
        sub_m['Date'] = pd.to_datetime(sub_m['Date'])

        TEST = TEST[['Date','PPI', 'CPI', 'Unemployment_Rate']]
        TEST['Date'] = pd.to_datetime(TEST['Date'])

        y = sub_m['Weekly Sales']
        X = sub_m.drop(['Weekly Sales'], axis = 1)
        X_train,X_test,y_train,y_test=X[:train_size],X[train_size:],y[:train_size],y[train_size:]

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


def champion_challenger(forecast_output1):    
    # Champion Challenger Framework
    champion = forecast_output1.drop_duplicates(subset = ["SKU"])[['SKU','Prophet_Error','Arima_Error','ETS_Error','RandomForest_Error','XGB_Error','Croston_Error']]
    champion = champion.set_index('SKU')

    Model = champion.idxmin(axis=1).reset_index()
    Model = pd.DataFrame(Model)

    y=pd.DataFrame()
    final=pd.DataFrame()
    final_output = pd.DataFrame()

    for i,k in Model.iterrows():

        sku_val=k['SKU']
        val1=k[0].split("_")[0]
        val=val1+"_Forecast"

        y['Champion_Model_Forecast'] = forecast_output1[forecast_output1['SKU']==sku_val][val].reset_index(drop = True)
        y['Error'] = forecast_output1[forecast_output1['SKU']==sku_val][val1+'_Error'].reset_index(drop = True)
        x = forecast_output1[forecast_output1['SKU']==sku_val][['Date','Brands','Categories','Region',
                                                                'Prophet_Forecast','Arima_Forecast', 'ETS_Forecast',
                                                                'RandomForest_Forecast','XGB_Forecast','Croston_Forecast']].reset_index(drop = True)
        final = pd.concat([y,x], axis = 1)

        final['Model'] = val1
        final['SKU'] = sku_val


        final_output=final_output.append(final)    

    cols2 = ['Date', 'Brands', 'Categories', 'Region','SKU', 'Model','Champion_Model_Forecast','Error',
             'Prophet_Forecast','Arima_Forecast', 'ETS_Forecast','RandomForest_Forecast','XGB_Forecast','Croston_Forecast']
    
    final_output = final_output[cols2].reset_index(drop = True)
    final_output
    
    return final_output

def main():
    train_data = pd.read_csv('SC_Train_Dataset.csv')
    test_data = pd.read_csv('SC_Test_dataset.csv')
    train_data = new_product_data_preparation(train_data)
    forecast = training_function(train_data,test_data)
    final_output = champion_challenger(forecast)
    final_output.to_csv('Champion_Model_Outputspy.csv', index = False)

if __name__=="__main__":
    main()

