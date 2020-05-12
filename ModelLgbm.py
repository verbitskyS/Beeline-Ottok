from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from Model import Model

file_name = 'lgbm_model.pkl'

class LGBM(Model):
    def __init__(self):
        super(LGBM, self).__init__(LGBMClassifier())


    def fit_atstart(self, X_train, y_train, params, save = False):
        self.model = LGBMClassifier(**params)
        print('Начало тренировки модели LGBM...')
        self.model.fit(X_train, y_train)
        print('Конец тренировки, сохранение модели LGBM...')
        if save: self.save_model(self.model, file_name)

    def fit_continue(self, X_train, y_train, save=False):
        self.model = self.load_model(file_name)
        print('Продолжение тренировки модели LGBM....')
        self.model.fit(X_train, y_train)
        print('Конец тренировки, сохранение модели LGBM...')
        if save: self.save_model(self.model, file_name)


    def save_model(self, file_name):
        self.model.save_model(file_name)


    def load_model(self, file_name):
        pass



