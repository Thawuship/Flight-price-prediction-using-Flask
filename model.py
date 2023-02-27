# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,MinMaxScaler
import pickle

df = pd.read_csv('flight.csv')

#Drop Unnamed value:
df.drop(columns = ['Unnamed: 0'],inplace = True)

cat_columns = ['airline','source_city','destination_city','class',]
num_columns = ['duration']
encoder = OrdinalEncoder().fit_transform(df[cat_columns])
encoder = pd.DataFrame(encoder,columns = cat_columns)

X = pd.concat([encoder,df[num_columns]],axis=1)
Y= df['price']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 42)

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, Y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[5, 1500, 2,4,2,4,5,1,2,1]]))