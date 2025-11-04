import numpy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_lr(df):
    '''
    Trains a linear regression model to predict coffee ratings 
    based on price per 100g.
        
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing coffee data, including 100g_USD and rating columns.
    
    Returns
    -------
    model_1 : sklearn linear regression model
        Trained linear regression model using 100g_USD to predict rating.
    '''
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
    '''
    Maps all different roast categories to numerical values.
    
    Parameters
    ----------
    roast : pandas.Series
        Series containing roast category strings from roast column in dataframe.
        
    Returns
    -------
    Mapped pandas.Series
        Series with roast categories mapped to numerical values to be applied to the roast column.
    '''
    roast_cat = {'Light' : 1, 
                 'Medium-Light' : 2, 
                 'Medium' : 3, 
                 'Medium-Dark' : 4, 
                 'Dark' : 5, 
                 numpy.nan : 0}
    return roast.map(roast_cat)

def train_dtr(df):
    '''
    Trains a decision tree regression model to predict coffee ratings 
    based on price per 100g and roast category.
        
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing coffee data, including 100g_USD, 
        roast, and rating columns.
    
    Returns
    -------
    model_2 : sklearn decision tree regression model
        Trained decision tree regression model using 100g_USD and 
        roast category to predict rating.
    '''
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    features = ['100g_USD', 'roast']
    
    # Map roast categories to numerical values
    df_train['roast'] = roast_category(df_train['roast'])
    
    X = df_train[features]
    y = df_train['rating']

    model_2 = DecisionTreeRegressor()
    model_2.fit(X, y)

    with open('./model_2.pickle', 'wb') as f:
        pickle.dump(model_2, f)
    
    return model_2

# Load data and train models when script is run
if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv'
    df = pd.read_csv(url)
    
    # Train and save model_1
    train_lr(df)
    
    # Train and save model_2
    train_dtr(df)


