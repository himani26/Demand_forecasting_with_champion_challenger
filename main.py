import uvicorn
import pandas as pd
import sys
from fastapi import FastAPI,HTTPException
from preprocess import new_product_data_preparation
from training import training_function
from champion import champion_challenger
import traceback


app = FastAPI(title="UAP Supply Chain - Demand forecasting Data Science APIs", 
             version = "0.1")

error_status_code = 400

@app.get("/V1/forecast/")
async def app_main():
    try:
        print('xyz')
        train_data = pd.read_csv('SC_Train_Dataset.csv')
        test_data = pd.read_csv('SC_Test_dataset.csv')
        train_data = new_product_data_preparation(train_data)
        forecast = training_function(train_data,test_data)
        final_output = champion_challenger(forecast)
        final_output.to_csv('Champion_Model_OutputSpyder.csv', index = False)
        #write_df() # writes results to db
        return {'message': 'success'}
    except:
        raise HTTPException(status_code=error_status_code, detail = f"Error : {sys.exc_info()}")
        trace_back = traceback.format_exc()
        return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





