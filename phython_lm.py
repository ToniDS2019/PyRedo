

# Load Dependences 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys 

#Load Data File


dataset = pd.read_csv(sys.argv[1])
print(dataset)

#Plot scatter of data and save as png file 

dataset.plot.scatter(x='x', y='y')
plt.savefig("dataset.png") 
plt.clf()


# In[4]:

pip install -U scikit-learn


# In[5]:

model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# In[6]:

Plot data with lienar fit and save as png file 

plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.savefig("linearmodel.png") 
plt.clf()







