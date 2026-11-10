import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("placement.csv")
df = df.iloc[:,1:]
plt.scatter(df["CGPA"],df["IQ"],c=df["Placement"])
plt.show()
x = df.iloc[:,0:2]
y = df.iloc[:,-1]
print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#model training
clf = LogisticRegression()
clf.fit(x_train,y_train )
clf.predict(x_test)


