import pandas as pd
import warnings
warnings.filterwarnings('ignore')


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