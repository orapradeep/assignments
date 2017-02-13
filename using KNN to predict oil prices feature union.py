import Quandl
import gzip
import simplejson as json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import FeatureUnion
from sklearn import linear_model
import urllib
import pickle
import dill
from sklearn.neighbors import KNeighborsRegressor


authtoken = None


def getQuandle(what):
    """ 
    Wrapper around Quandl requests, using authtoken only if available
    """
    if authtoken:
        return Quandl.get(what, authtoken=authtoken)
    else:
        return Quandl.get(what)

oil = getQuandle("DOE/RWTC")
print oil

oilm=oil.as_matrix()

for i in reversed(range(len(oilm))):
    oilm[i]=oilm[i]/oilm[i-10]
oil=pd.DataFrame(oilm)
PERIOD_MONTH = 20
for i in range(1,49):
    PREDICTION_LAG = i*PERIOD_MONTH*.5
    oil[i] = oil[0].shift(PREDICTION_LAG)



print oil


oilm=oil.as_matrix()
xL=oilm[1000:5000,1:49]
yL=oilm[1000:5000,0]
xT=oilm[5000:7000,1:49]
yT=oilm[5000:7000,0]


class LMNeighbors(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.p=KNeighborsRegressor(n_neighbors=1000,weights='distance',p=1)

    def fit(self, X, y):
        self.p.fit(X,y)
        return self

    def transform(self,X):
        return self.p.predict(X)

class LFNeighbors(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.p=KNeighborsRegressor(n_neighbors=500,weights='distance',p=1.2)

    def fit(self, X, y):
        self.p.fit(X,y)
        return self

    def transform(self,X):
        return self.p.predict(X)

class SMNeighbors(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.p=KNeighborsRegressor(n_neighbors=1000,weights='distance',p=1)

    def fit(self, X, y):
        X1=X[:,0:25]
        self.p.fit(X1,y)
        return self

    def transform(self,X):
        X1=X[:,0:25]
        return self.p.predict(X1)

class SFNeighbors(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.p=KNeighborsRegressor(n_neighbors=500,weights='distance',p=1.2)

    def fit(self, X, y):
        X1=X[:,0:25]
        self.p.fit(X1,y)
        return self

    def transform(self,X):
        X1=X[:,0:25]
        return self.p.predict(X1)

class OilLR(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.p=LinearRegression()

    def fit(self, X, y):
        self.p.fit(X,y)
        return self

    def transform(self,X):
        return self.p.predict(X)


class OilMod(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.fu=FeatureUnion([("a",LMNeighbors()),("b",LFNeighbors()),("c",SMNeighbors()),("d",SFNeighbors()),("e",OilLR())])
        self.lm=GridSearchCV(RandomForestRegressor(),{'n_estimators':[20,40],'max_depth':[2,4,8,16,32]})

    def fit(self,X,y):
        self.lm.fit(self.fu.fit_transform(X,y).reshape(5,len(X)).T,y)
        return self

    def predict(self, X):
        import pandas as pd
        import numpy as np
        return self.lm.predict(self.fu.transform(X).reshape(5,len(X)).T)






