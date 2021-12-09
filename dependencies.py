import pandas as pd
import numpy as np
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


