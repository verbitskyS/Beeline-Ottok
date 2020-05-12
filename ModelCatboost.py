from catboost import CatBoostClassifier
import catboost
from Model import Model
import shap


file_name = 'catboost_model.cbm'

class Catboost(Model):
    def __init__(self):
        super(Catboost, self).__init__(CatBoostClassifier())


    def fit_atstart(self, X_train, y_train, params, save = False):
        self.model = CatBoostClassifier(**params)
        print(self.model)
        print('Начало тренировки модели Catboost...')
        self.model.fit(X_train, y_train)
        print('Конец тренировки, сохранение модели Catboost....')
        self.save_model(file_name)

    def fit_continue(self, X_train, y_train, save = False):
        self.load_model(file_name)
        print('Продолжение тренировки модели Catboost...')
        self.model.fit(X_train, y_train)
        print('Конец тренировки, сохранение модели Catboost...')
        self.save_model(self.model, file_name)

    def save_model(self, file_name):
        self.model.save_model(file_name)


    def load_model(self, file_name):
        self.model = catboost.load_model(file_name, format='cbm')
