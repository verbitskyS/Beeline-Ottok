import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import time
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle


class DataLoader:
    def __init__(self, val_size=0.05, lgbm=True, save_train=False):
        self.val_size = val_size
        self.lgbm = lgbm
        self.all_train_X = None
        self.all_train_y = None
        self.all_train_X_ohe = None
        self.save_train = save_train

        self.test_data = None
        self.test_data_ohe = None

        self.all_val_X = None
        self.all_val_y = None
        self.all_val_X_ohe = None
        self.true_columns = ['Регион', 'Life_Time', 'Индикатор интеракций',
                             'Индикатор наличия мобильного договора на ИНН',
                             'Количество входящих звонков топовому оператору-конкуренту',
                             'Количество входящих звонков топовому оператору-конкуренту.1',
                             'Количество заходов на сайты с контактных номеров на топового оператора-конкурента',
                             'Количество заходов на сайты с фиксовых ip на топового оператора-конкурента',
                             'Количество исходящих звонков топовому оператору-конкуренту',
                             'Количество исходящих звонков топовому оператору-конкуренту.1', 'Предиктор по NPS',
                             'Предиктор по ТТ (рост на 15% и более)',
                             'Предиктор по блокировкам',
                             'Предиктор по исходящему data-трафику', 'Предиктор по исходящему голосовому трафику',
                             'Предиктор по отсутствию модификаций/подключений', 'Сегмент контракта',
                             'Топовый Оператор-конкурент по входящим звонкам контактных номеров',
                             'Топовый Оператор-конкурент по входящим звонкам фиксовых номеров',
                             'Топовый Оператор-конкурент по заходам на сайт контактных номеров',
                             'Топовый Оператор-конкурент по заходам на сайт фиксовых ip',
                             'Топовый Оператор-конкурент по исходящим звонкам контактных номеров',
                             'Топовый Оператор-конкурент по исходящим звонкам фиксовых номеров', 'Филиал', 'churn']

        self.cat_features = ['Регион', 'Филиал', 'Сегмент контракта',
                             'Топовый Оператор-конкурент по исходящим звонкам фиксовых номеров',
                             'Топовый Оператор-конкурент по входящим звонкам фиксовых номеров',
                             'Топовый Оператор-конкурент по исходящим звонкам контактных номеров',
                             'Топовый Оператор-конкурент по входящим звонкам контактных номеров',
                             'Топовый Оператор-конкурент по заходам на сайт фиксовых ip',
                             'Топовый Оператор-конкурент по заходам на сайт контактных номеров']

    def load_from_dir(self, path_data, path_real_churn, dates, test=False):
        self.dates = dates
        data = []
        data_real_churn_list = []
        for path in path_data:
            data.append(pd.read_excel(path))

        for path in path_real_churn:
            df = pd.read_csv(path)
            if 'Client_gen_id' in list(df):
                df = df.rename(columns={'Client_gen_id': 'id'})
                df = df.dropna(subset=['id'])
                df = df[df.id != 'Удалено по RFC 330477']
                df.id = df.id.astype('int64')
                df.id = df.id.astype('str')
            elif 'contract_no' in list(df):
                df = df.rename(columns={'contract_no': 'id'})
            else:
                print("Нету ID пользователя (или контракта)")
            df = df.loc[:, ['id', 'date_db']]
            data_real_churn_list.append(df)

        data_real_churn = pd.concat(data_real_churn_list)
        print('Данные загружены...\nНачало обработки')
        return self.prepare(data, data_real_churn, dates, test)

    def prepare(self, data, data_real_churn, dates, test=False):
        time_now = time.time()
        data_real_churn = data_real_churn.dropna(subset=['id'])
        data_real_churn.date_db = pd.to_datetime(data_real_churn.date_db)

        all_data = self.merge_train_data(data, data_real_churn, dates)

        all_data['Предиктор по исходящему голосовому трафику'] = all_data[
            'Предиктор по исходящему голосовому трафику'].apply(lambda x: 0 if x == 0 else 1)
        all_data['Предиктор по отсутствию модификаций/подключений'] = all_data[
            'Предиктор по отсутствию модификаций/подключений'].apply(lambda x: 0 if x == 0 else 1)
        all_data['Предиктор по блокировкам'] = all_data['Предиктор по блокировкам'].apply(lambda x: 0 if x == 0 else 1)
        all_data['Предиктор по ТТ (рост на 15% и более)'] = all_data['Предиктор по ТТ (рост на 15% и более)'].apply(
            lambda x: 0 if x == 0 else 1)
        all_data['Предиктор по NPS'] = all_data['Предиктор по NPS'].apply(lambda x: 0 if x == 0 else 1)
        all_data['Предиктор по исходящему data-трафику'] = all_data['Предиктор по исходящему data-трафику'].apply(
            lambda x: 0 if x == 0 else 1)
        all_data['Life_Time'] = all_data['Life_Time'].apply(lambda x: 2134 if x == 'No_data' else x)
        all_data['Предиктор по исходящему data-трафику'] = all_data['Предиктор по исходящему data-трафику'].apply(
            lambda x: 0 if x == 0 else 1)

        if self.save_train: all_data.to_csv('train_prepared.csv', index=False)

        if not test:
            all_data = pd.concat([all_data, all_data[all_data.churn == 1],all_data, all_data[all_data.churn == 1],
                                  all_data[all_data.churn == 1], all_data[all_data.churn == 1],
                                  all_data[all_data.churn == 1], all_data[all_data.churn == 1],
                                  all_data[all_data.churn == 1], all_data[all_data.churn == 1]
                                  ] )  # дублируем данные, чтобы классы стали более сбалнсированные
            self.all_train_X = all_data.drop(['churn'], axis=1)
            self.all_train_y = all_data.churn

            self.all_train_X, self.all_val_X, self.all_train_y, \
            self.all_val_y = train_test_split(self.all_train_X, self.all_train_y, test_size=self.val_size, random_state=42)

            if self.lgbm:
                self.all_train_X_ohe = self.one_hot_encoding(self.all_train_X, self.cat_features)
                self.all_val_X_ohe = self.one_hot_encoding(self.all_val_X, self.cat_features, test=True)

        else:
            all_data = all_data.drop(['churn'], axis = 1)  #тут он все равно нулевой, тк мы не можем заглянуть в будущее
            self.test_data = all_data
            if self.lgbm:
                print(self.test_data.shape)
                self.test_data_ohe = self.one_hot_encoding(self.test_data, self.cat_features, test=True)

        del all_data
        gc.collect()
        print('Конец обработки')
        if not test:
            print('Размер тренировчных: {}'.format(self.all_train_X.shape))
            print('Размер валидационных: {}'.format(self.all_val_X.shape))
            print('Размер: {}'.format(sys.getsizeof(self.all_train_X) + sys.getsizeof(self.all_train_X_ohe)))
        else:
            print('Размер тренировчных: {}'.format(self.test_data.shape))
            print('Размер: {}'.format(sys.getsizeof(self.test_data) + sys.getsizeof(self.test_data_ohe)))
        print('Затрачено времени: {}'.format(time.time() - time_now))

    def one_hot_encoding(self, data, cat_features, test=False):
        data_ohe_without_cat = data.drop(cat_features, axis=1).to_numpy()

        if not test:
            ohe = OneHotEncoder(handle_unknown='ignore')
            data_ohe = ohe.fit_transform(data[cat_features]).toarray()
            data_ohe = np.hstack([data_ohe_without_cat, data_ohe])
            with open("encoder.pickle", "wb") as f:
                pickle.dump(ohe, f)
            print("Размеры после OHE = " + str(data_ohe.shape))

        else:
            with open('encoder.pickle', 'rb') as f:
                ohe = pickle.load(f)
            data_ohe = ohe.transform(data[cat_features]).toarray()
            data_ohe = np.hstack([data_ohe_without_cat, data_ohe])

        return data_ohe

    def load_from_db(self):
        pass

    def merge_train_data(self, data, data_real_churn, dates):
        for i in range(len(data)):
            if 'Client_gen_id' in list(data[i]):
                data[i] = data[i].rename(columns={'Client_gen_id': 'id'})
            elif 'Contract_NO' in list(data[i]):
                data[i] = data[i].rename(columns={'Contract_NO': 'id'})
            else:
                print(list(data[i]))
                print("Нету ID пользователя (или контракта)")

            if 'Предиктор по интеракциям/заданиям за 3 месяца' in list(data[i]):
                data[i] = data[i].rename(
                    columns={'Предиктор по интеракциям/заданиям за 3 месяца': 'Предиктор по интеракциям/заданиям'})

            data[i].id = data[i].id.astype('str')
            data[i] = data[i].set_index('id')
            train_indeces = list(data_real_churn[(data_real_churn.date_db > dates[i])].id)
            data[i]['churn'] = 0
            data[i].loc[list(set(data[i].index) & set(train_indeces)), 'churn'] = 1

            print(list(set(list(data[i])) - set(self.true_columns)))
            data[i] = data[i].drop(list(set(list(data[i])) - set(self.true_columns)), axis=1)  #выбрасываем лишние стобцы

            data_real_churn['churn'] = 1
            # Новая фича - количество оттоков каждого пользователя раннее -- скор выше гораздо
            ammounts_churn = data_real_churn[data_real_churn.date_db < dates[i]].groupby('id').churn.count()
            data[i]['churns_early'] = 0
            data[i].loc[list(set(ammounts_churn.index) & set(data[i].index)), 'churns_early'] \
                = ammounts_churn.loc[list(set(ammounts_churn.index) & set(data[i].index))]

            # Новая фича - количество дней после последнего оттока по какой либо услуге раннее -- скор выше
            last_churn = data_real_churn[data_real_churn.date_db < dates[i]].groupby('id').date_db.max()
            last_churn = (pd.to_datetime(dates[0]) - last_churn).dt.days
            data[i]['last_churn'] = 0
            data[i].loc[list(set(last_churn.index) & set(data[i].index)), 'last_churn'] \
                = last_churn.loc[list(set(last_churn.index) & set(data[i].index))]

        all_data = pd.concat(data)
        # else:
        #   print("ОШИБКА, ДАННЫЕ НЕВЕРНЫЕ, НЕ ХВАТАЕТ СТОЛБЦОВ: " + str(set(self.true_columns) - set(list(data[i]))))


        print('Churn = ' + str(all_data.churn.mean() * 100) + '%')

        # all_data = all_data.merge(train_churn, left_on='Client_gen_id', right_on='Client_gen_id', how='left')

        del data, data_real_churn
        gc.collect()

        return all_data