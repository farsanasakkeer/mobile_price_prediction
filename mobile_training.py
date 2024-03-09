import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


df=pd.read_csv('mobile_rate.csv')

df["Ram"].fillna(value=0, inplace = True)
df["Rom"].fillna(value=0, inplace = True)
df["Extended_Memory"].fillna(value=0, inplace = True)
df["Battery_capacity(mAh)"].fillna(value=df['Battery_capacity(mAh)'].mean(), inplace = True)
df["Processor"].fillna("No processor", inplace = True)





# Split-out validation dataset
from sklearn.model_selection import train_test_split

array = df.values
x= array[:,1:5]  
y = array[:,6] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)

from sklearn.linear_model import Lasso
classifier = Lasso().fit(x_train, y_train)
print(classifier.predict([[6.0,64.000,512.0,6000.0]]))

# Creating a pickle file for the classifier
filename = 'mobile_rate_pred.pkl'
pickle.dump(classifier, open(filename, 'wb'))