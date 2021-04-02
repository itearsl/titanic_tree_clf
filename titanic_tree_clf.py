from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

params = {
    'n_estimators': range(10, 20, 3),
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 9),
    'min_samples_split': range(2, 10, 3),
    'min_samples_leaf': range(1, 15, 2),
}

data = pd.read_csv('train.csv')
data.head()

data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
data = data.sample(frac=1)
data.info()

X_data = data.drop(['Survived'], axis=1)
y_data = data['Survived']
X_data = pd.get_dummies(X_data)
mean_age = X_data.Age.mean()
X_data.fillna(mean_age, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)
X_test.head()

rf_clf = RandomForestClassifier()
rf_grid_search_cv = GridSearchCV(rf_clf, params, cv=4)
rf_grid_search_cv.fit(X_train, y_train)
rf_best_clf = rf_grid_search_cv.best_estimator_
y_pred = rf_best_clf.predict(X_test)
print(rf_grid_search_cv.best_params_)
print(y_pred)

acc = rf_best_clf.score(X_test, y_test)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

metrics = [acc, prec, rec]
metrics

test_data = pd.read_csv('test.csv')

test_data.head()

submission = test_data.PassengerId
test_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

test_data= pd.get_dummies(test_data)

pred = rf_best_clf.predict(test_data)
submission = pd.DataFrame({"PassengerId":submission, 'Survived':pred})

submission.to_csv('submission.csv', index=False)