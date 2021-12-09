import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

