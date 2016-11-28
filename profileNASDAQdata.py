import pandas as pd
import numpy as np

data = pd.read_csv("/Users/tabish/MLProject/AAPL_stock_data.csv", index_col="Date")
columns = data.columns.values
print columns

for column in columns:
        if column != "Date":
                print column, data[column].max(), data[column].min(), data[column].mean()
data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
X = (data.index - data.index[0]).days.reshape(-1,1)
print data.index
print X
y = data["Close"].values
print y

from sklearn.neighbors import KNeighborsRegressor
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#X = []
#y = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#kf = KFold(n_splits=2, random_state=None, shuffle=False)
knn = KNeighborsRegressor()
lr = KernelRidge()
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
krGrid = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
krGrid.fit(X_train, y_train)

print "KNN Score = ", knn.score(X_test, y_test)
print "Kernel Ridge = ", lr.score(X_test, y_test)
print krGrid.score(X_test, y_test)
print grid_search.cv_results_
'''
for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
'''
#print cross_val_score(kf, X, y)
#print cross_val_score(lr, X, y)

