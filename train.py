import numpy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_lr(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    features = ['100g_USD']

    X = df_train[features]
    y = df_train['rating']

    model_1 = LinearRegression()
    model_1.fit(X, y)

    with open('./model_1.pickle', 'wb') as f:
        pickle.dump(model_1, f)
            
    return model_1

def roast_category(roast):
    roast_cat = {'Light' : 1, 
                 'Medium-Light' : 2, 
                 'Medium' : 3, 
                 'Medium-Dark' : 4, 
                 'Dark' : 5, 
                 numpy.nan : 0}
    return roast.map(roast_cat)

def train_dtr(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(df_train.columns)
    if 'roast' in df_train.columns:
        features = ['100g_USD', 'roast']
        df_train['roast_cat'] = roast_category(df_train['roast'])

    else:
        features = df_train.columns
        df_train['roast_cat'] = df_train['roast_cat'].fillna(0).astype(int)

    X = df_train[features]
    y = df_train.values

    model_2 = DecisionTreeRegressor()
    model_2.fit(X, y)

    with open('./model_2.pickle', 'wb') as f:
        pickle.dump(model_2, f)
    
    return model_2

