# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
path = "C:/Users/300997447"
filename = 'Advertising.csv'
fullpath = os.path.join(path,filename)
data_mayy_adv = pd.read_csv(fullpath)
data_mayy_adv.columns.values
data_mayy_adv.shape
data_mayy_adv.describe()
data_mayy_adv.dtypes
data_mayy_adv.head(5)
###########


import numpy as np
def corrcoeff(df,var1,var2):
    df['corrn']=(df[var1]-np.mean(df[var1]))*(df[var2]-np.mean(df[var2]))
    df['corrd1']=(df[var1]-np.mean(df[var1]))**2
    df['corrd2']=(df[var2]-np.mean(df[var2]))**2
    corrcoeffn=df.sum()['corrn']
    corrcoeffd1=df.sum()['corrd1']
    corrcoeffd2=df.sum()['corrd2']
    corrcoeffd=np.sqrt(corrcoeffd1*corrcoeffd2)
    corrcoeff=corrcoeffn/corrcoeffd
    return corrcoeff
print(corrcoeff(data_mayy_adv,'TV','Sales'))
print(corrcoeff(data_mayy_adv,'Radio','Sales'))
print(corrcoeff(data_mayy_adv,'Newspaper','Sales'))
#################

import matplotlib.pyplot as plt
plt.plot(data_mayy_adv['TV'],data_mayy_adv['Sales'],'ro')
plt.title('TV vs Sales')
plt.plot(data_mayy_adv['Radio'],data_mayy_adv['Sales'],'ro')
plt.title('Radio vs Sales')
plt.plot(data_mayy_adv['Newspaper'],data_mayy_adv['Sales'],'ro')
plt.title('Newspaper vs Sales')

## Predicte a new value
import statsmodels.formula.api as smf
model1=smf.ols(formula='Sales~TV',data=data_mayy_adv).fit()
model1.params
#################
print(model1.pvalues)
print(model1.rsquared)
print(model1.summary())


import statsmodels.formula.api as smf
model3=smf.ols(formula='Sales~TV+Radio',data=data_mayy_adv).fit()
print(model3.params)
print(model3.rsquared)
print(model3.summary())
## Predicte a new value
X_new2 = pd.DataFrame({'TV': [50],'Radio' : [40]})
# predict for a new observation
sales_pred2=model3.predict(X_new2)
print(sales_pred2)



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
feature_cols = ['TV', 'Radio']
X = data_mayy_adv[feature_cols]
Y = data_mayy_adv['Sales']
trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(trainX, trainY)
print (lm.intercept_)
print (lm.coef_)
zip(feature_cols, lm.coef_)
[('TV', 0.045706061219705982), ('Radio', 0.18667738715568111)]
lm.score(trainX, trainY)
lm.predict(testX)

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
feature_cols = ['TV', 'Radio','Newspaper']
X = data_mayy_adv[feature_cols]
Y = data_mayy_adv['Sales']
estimator = SVR(kernel="linear")
selector = RFE(estimator,2,step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.ranking_)
