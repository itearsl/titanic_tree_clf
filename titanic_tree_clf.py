from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 50)
}

data = pd.read_csv('train.csv')
data.head()

data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_data = data.drop(['Survived'], axis=1)
y_data = data['Survived']
X_data = pd.get_dummies(X_data)
mean_age = X_data.Age.mean()
X_data.fillna(mean_age, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)
X_test.head()

clf = DecisionTreeClassifier(())
grid_search_cv = GridSearchCV(clf, params, cv=5)
grid_search_cv.fit(X_train, y_train)
best_clf = grid_search_cv.best_estimator_

y_pred = best_clf.predict(X_test)

acc = best_clf.score(X_test, y_test)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)