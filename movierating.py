import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('IMDB India Movies.csv')

data['genre'] = data['genre'].str.split(',').str[0]
data['director'] = data['director'].fillna('Unknown')
data['actors'] = data['actors'].fillna('Unknown')
data['rating'] = data['rating'].astype(float)

features = ['genre', 'director', 'actors']
X = data[features]
y = data['rating']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

plt.figure(figsize=(10, 5))
sns.histplot(y_test - y_pred, bins=30, kde=True)
plt.title('Residuals Distribution')
plt.show()

print('RMSE:', rmse)
