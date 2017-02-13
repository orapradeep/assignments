from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np

class FlippingACoinEstimator(BaseEstimator, ClassifierMixin):


    def __init__(self):
        self.prob=0
        
    def fit(self, X, y):
        self.prob=float(sum(y))/len(y)

    def predict(self,X):
        our_results = np.random.rand(len(X),1)
        for i in range(len(our_results)):
            if our_results[i]<self.prob:
                our_results[i]=1
            else:
                our_results[i]=0
        return our_results
    

m=95
n=90
#Information about the company
inptL=np.random.rand(m,n)
#Stock Collapsed
outptL=np.random.randint(2, size=(m, 1))
inptT=np.random.rand(m,n)
outptT=np.random.randint(2, size=(m, 1))
ourCoin=FlippingACoinEstimator()

ourCoin.fit(inptL,outptL)
print ourCoin.predict(inptT)
print ourCoin.score(inptT,outptT)

print ourCoin.prob