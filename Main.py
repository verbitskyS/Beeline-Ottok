import DataPrepare
import ModelCatboost
import ModelLgbm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import shap
cat_features = ['Регион', 'Филиал', 'Сегмент контракта',
                'Топовый Оператор-конкурент по исходящим звонкам фиксовых номеров',
                'Топовый Оператор-конкурент по входящим звонкам фиксовых номеров',
                'Топовый Оператор-конкурент по исходящим звонкам контактных номеров',
                'Топовый Оператор-конкурент по входящим звонкам контактных номеров',
                'Топовый Оператор-конкурент по заходам на сайт фиксовых ip',
                'Топовый Оператор-конкурент по заходам на сайт контактных номеров']

best_params_catboost = {'n_estimators': 100,
                             'cat_features':cat_features, 'verbose': 0}

best_params_lgbm = {'num_iterations': 300, 'max_depth': 5}

params_catboost_cv = {'depth': [3, 6, 4, 5, 7, 8, 9, 10],
                      'n_estimators': [250, 50, 100, 500, 1000],
                      'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                      'l2_leaf_reg': [3, 1, 5, 10, 100],
                      'border_count': [50, 5, 10, 20],
                      'ctr_border_count': [50, 5, 10, 20, 100, 200],
                      'thread_count': 4, 'cat_features': [cat_features], 'verbose': [0]}

params_lgbm_cv = {'max_depth': [3, 6, 4, 5, 7, 8, 9, 10],
                  'num_iterations': [250, 50, 100, 500, 1000],
                  'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                  'lambda_l2': [3, 1, 5, 10, 100],
                  'bagging_fraction': [0.5, 0.75, 0.9],
                  'bagging_freq': [0.5, 0.75, 0.9],
                  'num_leaves': [50, 5, 10, 20],
                  'max_bin': [50, 5, 10, 20, 100, 200]}


def ensemble_predict( models, weights, X_tests):
    proba_ensemble = np.array([models[i].predict_proba(X_tests[i]) * weights[i] for i in len(models)]).sum(axis=0)
    predict_ensemble = np.round(proba_ensemble)
    return proba_ensemble, predict_ensemble


def evalute_test(y_test, y_pred, y_probs):
    print('f1 = {} \nprecision = {} \nrecall = {} \nroc_auc = {}'
          .format(f1_score(y_test, y_pred), precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred), roc_auc_score(y_test, y_probs)))


def load_data_for_models(val_size=0.2, lgbm=True):
    dataLoader = DataPrepare.DataLoader(val_size, lgbm)
    dataLoader.load_from_dir(path_data=['train_data/september.xlsx', 'train_data/october.xlsx', 'train_data/april1.xlsx', 'train_data/april1.xlsx'],
                                   path_real_churn=['train_data/real_churn_old.csv', 'train_data/real_churn_new.csv'],
                             dates=['2019-08-31', '2019-10-01', '2020-04-01', '2020-04-14'])
    X_train = dataLoader.all_train_X
    y_train = dataLoader.all_train_y
    X_val = dataLoader.all_val_X
    y_val = dataLoader.all_val_y
    print(y_train.value_counts(normalize=True))
    if lgbm:
        X_train_ohe = dataLoader.all_train_X_ohe
        X_val_ohe = dataLoader.all_val_X_ohe
        return X_train, y_train, X_val, y_val, X_train_ohe, X_val_ohe
    else: return X_train, y_train, X_val, y_val, None


def load_test(test_data_path, date, lgbm=True):
    dataLoader = DataPrepare.DataLoader(val_size=0, lgbm = lgbm)
    dataLoader.load_from_dir(path_data=[test_data_path],
                                   path_real_churn=['train_data/real_churn_old.csv', 'train_data/real_churn_new.csv'], dates=date, test=True)
    X_test = dataLoader.test_data
    if lgbm:
        X_test_ohe = dataLoader.test_data_ohe
        return X_test, X_test_ohe
    else: return X_test, None



class Model:
    def __init__(self, w=0.5, val_size=0.1, lgbm=True, validation=True):
        self.val_size = val_size
        self.lgbm = lgbm
        self.validation = validation
        self.model_catboost = None
        self.model_lgbm = None
        self.w = w

    def train(self, feature_importance=False):
        X_train, y_train, X_val, y_val, X_val_ohe, X_test_ohe = load_data_for_models(val_size=self.val_size, lgbm=self.lgbm )
        self.model_catboost = ModelCatboost.Catboost()
        self.model_catboost.fit_atstart(X_train, y_train, best_params_catboost, save=False)
        if self.lgbm :
            self.model_lgbm = ModelLgbm.LGBM()
            self.model_lgbm.fit_atstart(X_val_ohe, y_train, best_params_lgbm, save=False)

        if self.validation:
            catboost_proba = self.model_catboost.predict_proba(X_val)
            catboost_preds = self.model_catboost.predict(X_val)
            print('CATBOOST')
            evalute_test(y_val, catboost_preds, catboost_proba)
            if self.lgbm:
                lgbm_proba = self.model_lgbm.predict_proba(X_test_ohe)
                lgbm_proba = self.model_lgbm.predict_proba(X_test_ohe)
                catboost_preds = self.model_catboost.predict(X_val)
                lgbm_preds = self.model_lgbm.predict(X_test_ohe)
                print("LGBM")
                evalute_test(y_val, lgbm_preds, lgbm_proba)

                for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    ensemble_proba = catboost_proba * w + lgbm_proba * (1 - w)
                    ensemble_preds = np.round(ensemble_proba)
                    print('\nw = {}'.format(w))
                    evalute_test(y_val, ensemble_preds, ensemble_proba)


            if feature_importance:
                shap_test = shap.TreeExplainer(self.model_catboost.model).shap_values(X_train)
                shap.summary_plot(shap_test, X_train, max_display=10, auto_size_plot=True)




    def test(self, test_data_path, dates):
        X_test, X_test_ohe = load_test(test_data_path, dates, lgbm=self.lgbm)
        preds_cat = self.model_catboost.predict_proba(X_test)
        if self.lgbm:
            preds_lgbm = self.model_lgbm.predict_proba(X_test_ohe)
            preds = self.w * preds_cat + self.w * preds_lgbm
        else:
            preds = [preds_cat]

        X_test['Churn'] = preds


        return X_test.loc[:, ['Регион', 'Филиал','Churn']]


