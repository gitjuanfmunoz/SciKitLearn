import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
rl=linear_model.LinearRegression()

data=pd.read_csv('movies.csv')
df=pd.DataFrame(data)
x=df['movie_facebook_likes']
y=df['imdb_score']

X=x[:,np.newaxis]
print(X)
print(rl.fit(X,y))
print(rl.coef_)
m=rl.coef_[0]
b=rl.intercept_
y_p=m*X+b #Valor Predice
print('y={0}*x+{1}'.format(m,b))
print(rl.predict(X)[0:5])
print("El valor de r^2 =",r2_score(y,y_p))
plt.scatter(x,y,color='blue')
plt.plot(x,y_p,color='red')
plt.title('Linear Regression', fontsize=18,color='black')
plt.xlabel('Likes FB', fontsize=12)
plt.ylabel('Imdb Score', fontsize=12)
plt.show()