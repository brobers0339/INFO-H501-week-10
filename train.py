import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

class Train_Models:

    def __init__(self, url):
        self.coffee_df = pd.read_csv(url)
        self.df_train, self.df_test = train_test_split(self.coffee_df, test_size=0.2, random_state=42)
        self.roast_cat = {}
        
    def train_LR(self):
        features = ['100g_USD']

        X = self.df_train[features]
        y = self.df_train['rating']

        model_1 = LinearRegression()
        model_1.fit(X, y)

        with open('./model_1.pickle', 'wb') as f:
            pickle.dump(model_1, f)
            
    def train_DTR(self):
        features = ['100g_USD', 'roast']

        X = self.df_train[features]
        y = self.df_train['rating']

        model_2 = DecisionTreeRegressor()
        model_2.fit(X, y)

        with open('./model_2.pickle', 'wb') as f:
            pickle.dump(model_2, f)
