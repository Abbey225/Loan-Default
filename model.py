import numpy as np
import pandas as pd

df=pd.read_csv('loandefault.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())

from imblearn.over_sampling import SMOTE
sm=SMOTE()
x=df.drop('Defaulted?',axis=1)
y=df['Defaulted?']

x_sam,y_sam=sm.fit_resample(x,y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_sam,y_sam,test_size=0.3,random_state=100,shuffle=True)

from sklearn.ensemble import ExtraTreesClassifier
et_model=ExtraTreesClassifier()

et_model.fit(x_train,y_train)

import pickle
pickle.dump(et_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
