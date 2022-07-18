import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 100
pd.options.display.max_columns = 20

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error

raw_all = pd.read_csv('bike-sharing-demand/Full.csv')

def non_feature_engineering(raw):
    if 'datetime' in raw_all.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'])
        raw['DateTime'] = pd.to_datetime(raw['datetime'])

    if raw.index.dtype == 'int64':
        raw.set_index('DateTime', inplace = True)

    raw = raw.asfreq('H', method = 'ffill')

    raw_nfe = raw.copy()
    return raw_nfe

def feature_engineering(raw):
    if 'datetime' in raw_all.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'])
        raw['DateTime'] = pd.to_datetime(raw['datetime'])

    if raw.index.dtype == 'int64':
        raw.set_index('DateTime', inplace=True)

    raw = raw.asfreq('H', method = 'ffill')

    result = sm.tsa.seasonal_decompose(raw['count'])
    Y_trend = pd.DataFrame(result.trend)
    Y_trend.fillna(method = 'ffill', inplace = True)
    Y_trend.fillna(method = 'bfill', inplace = True)
    Y_trend.columns = ['count_trend']
    Y_seasonal = pd.DataFrame(result.seasonal)
    Y_seasonal.fillna(method = 'ffill', inplace = True)
    Y_seasonal.fillna(method = 'bfill', inplace = True)
    Y_seasonal.columns = ['count_seasonal']

    if 'count_trend' not in raw:
        if 'count_seasonal' not in raw:
            raw = pd.concat([raw, Y_trend, Y_seasonal], axis = 1)

    Y_day = raw[['count']].rolling(24).mean()
    Y_day.fillna(method = 'ffill', inplace = True)
    Y_day.fillna(method = 'bfill', inplace = True)
    Y_day.columns = ['count_day']
    Y_week = raw[['count']].rolling(24*7).mean()
    Y_week.fillna(method = 'ffill', inplace = True)
    Y_week.fillna(method = 'bfill', inplace = True)
    Y_week.columns = ['count_week']

    if 'count_day' not in raw.columns:
        if 'count_week' not in raw.columns:
            raw = pd.concat([raw, Y_day, Y_week], axis = 1)

    Y_diff = raw[['count']].diff()
    Y_diff.fillna(method = 'ffill', inplace = True)
    Y_diff.fillna(method = 'bfill', inplace = True)
    Y_diff.columns = ['count_diff']

    if 'count_diff' not in raw.columns:
            raw = pd.concat([raw, Y_diff], axis = 1)

    raw['temp_group'] = pd.cut(raw['temp'], 10)

    raw['Year'] = raw.datetime.dt.year
    raw['Quarter'] = raw.datetime.dt.quarter
    raw['Quarter_ver2'] = raw['Quarter'] + (raw.Year - raw.Year.min()) * 4
    raw['month'] = raw.datetime.dt.month
    raw['day'] = raw.datetime.dt.day
    raw['Hour'] = raw.datetime.dt.hour
    raw['DayofWeek'] = raw.datetime.dt.dayofweek

    raw['count_lag1'] = raw['count'].shift(1)
    raw['count_lag1'].fillna(method = 'bfill', inplace = True)
    raw['count_lag2'] = raw['count'].shift(2)
    raw['count_lag2'].fillna(method = 'bfill', inplace = True)

    raw = pd.concat([raw, pd.get_dummies(raw['Quarter'], prefix = "Quarter_Dummy", drop_first = True)], axis = 1)

    raw_fe = raw.copy()
    return raw_fe

def datasplit_cs(raw, Y_colname, X_colname, test_size, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(raw[X_colname], raw[Y_colname], test_size = test_size,
                                                        random_state = random_state)

    return X_train, X_test, Y_train, Y_test

def datasplit_ts(raw, Y_colname, X_colname, criteria):
    raw_train = raw.loc[raw.index < criteria]
    raw_test = raw.loc[raw.index >= criteria]

    X_train = raw_train[X_colname]
    X_test = raw_test[X_colname]
    Y_train = raw_train[Y_colname]
    Y_test = raw_test[Y_colname]

    return X_train, X_test, Y_train, Y_test

raw_fe = feature_engineering(raw_all)
Y_colname = ['count']
X_remove = ['datetime', 'DateTime', 'temp_group', 'registered', 'casual']
X_colname = [x for x in raw_fe.columns if x not in Y_colname+X_remove]

X_train, X_test, Y_train, Y_test = datasplit_ts(raw_fe, Y_colname, X_colname, '2012-07-01')
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

fit_reg1 = sm.OLS(Y_train, X_train).fit()
pred_tr_reg1 = fit_reg1.predict(X_train).values
pred_tr_reg2 = fit_reg1.predict(X_test).values

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
raw_fe.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size':'15pt'})

def evaluation(Y_real, Y_pred, graph_on=False):
    loss_length = len(Y_real.values.flatten()) - len(Y_pred)
    if loss_length != 0:
        Y_real = Y_real[graph_on:]
    if graph_on == True:
        pd.concat([Y_real, pd.DataFrame(Y_pred, index = Y_pred.index, columns = ['prediction'])], axis = 1). plot(kind = 'line', figsize = (20,6), xlim = (Y_real.index.min(), Y_real.index.max()),
                                                                                                                  linewidth = 3, fontsize = 20)
        plt.title('Time Series of Target', fontsize = 20)
        plt.xlabel('Index', fontsize = 15)
        plt.ylabel('Targer Value', fontsize = 15)

    MAE = abs(Y_real.values.flatten() - Y_pred).mean()
    MSE = ((Y_real.values.flatten() - Y_pred)**2).mean()
    MAPE = (abs(Y_real.values.flatten() - Y_pred)/Y_real.values.flatten()*100).mean()

    Score = pd.DataFrame([MAE, MSE, MAPE], index = ['MAE', 'MSE', 'MAPE'], columns = ['Score']).T
    Residual = pd.DataFrame(Y_real.values.flatten() - Y_pred, index = Y_real.index, columns = ['Error'])

    return Score, Residual

def evaluation_trte(Y_real_tr, Y_pred_tr, Y_real_te, Y_pred_te, graph_on=False):
    Score_tr, Residual_tr = evaluation(Y_real_tr, Y_pred_tr, graph_on = graph_on)
    Score_te, Residual_te = evaluation(Y_real_te, Y_pred_te, graph_on = graph_on)
    Score_trte = pd.concat['Train', 'Test']
    Score_trte.index = ['Train', 'Test']

    return Score_trte, Residual_tr, Residual_te
