from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


class Model:

    def __init__(self, model):
        self.model = model


    def search_params(self, X_train, y_train, params, cv=5, scorings='roc_auc'):
        GSCV = GridSearchCV(estimator=self.model,
                            param_grid=params, scoring=scorings, cv=cv, n_jobs=-1, verbose=1)
        GSCV.fit(X_train, y_train)
        print(GSCV.best_score_)
        print(GSCV.best_params_)
        self.model = GSCV.best_estimator_

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


    def test_validation(self, X_train, y_train, cv=5):
        scoring = {'f1': 'f1',
                   'prec': 'precision',
                   'rec': 'recall',
                   'roc_auc': 'roc_auc'}

        scores = cross_validate(self.model, X_train, y_train, scoring=scoring, cv=cv)
        print('{}: \nf1 = {} \nprecision = {} \nrecall = {} \nroc_auc = {}'
              .format(str(type(self.model)).split('.')[-1], scores['test_f1'].mean(), scores['test_prec'].mean(),
                      scores['test_rec'].mean(), scores['test_roc_auc'].mean()))
