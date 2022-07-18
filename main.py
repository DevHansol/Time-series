# Ignore the Warnings
import warnings
# warnings.filterwarnings('always') 항상 경고문이 뜨게 함
warnings.filterwarnings('ignore') # 경고문이 안 뜨게 함

# System related and data input controls
import os # 윈도우, Rinux 폴더에 위치, 경로등을 다룰 수 있음

# Data manipulation and visualization
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format # 내가 입력한 데이터가 무엇이든 소수점 둘 째 자리까지 나타내 주세요
pd.options.display.max_rows = 100 # 내가 입력하는 데이터의 행이 몇 개가 있든지 딱 100개만큼 나타내 주세요
pd.options.display.max_columns = 20 # 내가 입력하는 데이터의 열이 몇 개가 있든지 딱 20만틈 나타내 주세요

import numpy as np
import matplotlib.pyplot as plt # 시각화를 담당하는 모듈
import seaborn as sns # 시각화를 담당하는 모듈

# Modeling algorithms
# General
import statsmodels.api as sm # 모든 통계 모듈을 가져옴 --> 이것을 사용해서 회귀분석이나 시계열 분석 라이브러리를 접근할 수 있음
from scipy import stats # 통계 검정 모듈 패키지

# Model selection
from sklearn.model_selection import train_test_split # model_selection에서 train과 test를 분리하는 모듈을 사용한다

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error

raw_all = pd.read_csv("bike-sharing-demand/Full.csv")

# raw_all.shape 현재 데이터프레임에 행과 열의 갯수가 출력됨
# raw_all.ndim 현재 데이터프레임이 몇 차원인지 출력함
# raw_all.head() 데이터프레임 상단에 5개만 출력
# raw_all.tail() 데이터프레임 하단에 5개만 출력
# raw_all.describe(include = 'all').T 데이터프레임에 통계를 내어 갯수를 표시한다
# raw_all.info() 데이터프레임의 데이터가 Type을 알려준다

'''
'datetime' in raw_all.columns 
datetime은 문자열로 저장되어 있는데 이를 pd.to_datetime() 명령어를 사용하여 컴퓨터가 이해할 수 있는 시간으로 나타내는 냄
'''

if 'datetime' in raw_all.columns:
    raw_all['datetime'] = pd.to_datetime(raw_all['datetime'])
    raw_all['DateTime'] = pd.to_datetime(raw_all['datetime'])

# raw_all.index 인덱스 명령어는 인덱스의 처음과 끝 즉 데이터의 갯수를 나타내고 시계열 분석에서 인덱스를 시간으로 바꾸는게 중요함
'''
인덱스를 시계열로 바꾸는 방법
1. raw_all.index.dtype 으로 인덱스의 데이터 타입을 확인한다.
2. raw_all.set_index('DateTime', inplace = True)를 활용하여 시계열로 바꾸면 됨
tip raw_all.set_index 안에 inplace = false로 설정하면 겉모양 바뀌고 인덱스는 시계열로 바뀌지 않음

bring back
if raw_all.index.dtype == 'int64':
    raw_all.reset_index(drop = False, inplace = True) 
'''

if raw_all.index.dtype == 'int64':
    raw_all.set_index('DateTime', inplace = True)

# raw_all.describe(include = 'all').T
# raw_all.isnull() 실제 데이터프레임 안에 null 값이 있는지 확인하는 명령어
# raw_all.isnull().sum() isnull() 명령어 사용시 실제 데이터프레임안에 null 값을 확인하기 어려워 주로 이 명령어를 사용함

#-------------------요기까지 데이터를 준비하는 단계--------------------------------------

# --------------------실제 frequency를 세팅하는 단계----------------------------------
'''
처음 준비 단계에서 인덱스를 시계열로 바꿀 때 이것이 year, month, weekend가 무작위로 설정이됨
그래서 raw_all.asfreq('H')[raw_all.asfreq('H').isnull().sum(axis = 1) > 0]으로 빈도를 달로 수정해야 함
여기서 raw_all.asfreq('H')로 수정하면 인덱스 시계열의 빈도는 설정되지만 다른 raw에서 nan값이 발생함
그래서 [raw_all.asfreq('H').isnull().sum(axis = 1) > 0]으로 빠져 있는 raw가 무엇인지 체크해줌
근데 axis = 1을 안 넣으면 에러가 남 --> 왜인지는 모르겠음
'''

raw_all.asfreq('H')[raw_all.asfreq('H').isnull().sum(axis = 1) > 0]
raw_all = raw_all.asfreq('H', method = 'ffill') # 앞에 있는 데이터를 nan 값을 채움
'''
ffill, bfill을 사용할 때 주의할 점은 컴퓨터는 데이터가 오름차순인지 내림차순인지 알 수 없다
그렇기에 시계열이 오름차순이면 ffill을 사용하고, 내림차순이면 bfill을 사용한다
'''

# raw_all.index
# raw_all.asfreq('D') 하루단위로 시계열을 바꿈
# raw_all.asfreq('W') 일주일 단위로 데이터를 변경
# raw_all.asfreq('H') 시간 단위로 데이터를 변경
# raw_all.asfreq('H').isnull().sum()
# raw_all.asfreq('H')[raw_all.asfreq('H').sum(axis = 1) > 0]
# raw_all.asfreq('H').head(100)

'''
raw_all에서 y의 후보인 count, registered, casual을 가지고 와서, 시각화(plot)를 하는데 형식은 선(line)을 그리고
시각화 그림 사이즈(figsize)는 20,6의 사이즈로, 선의 두께(linewidth)으로, 글씨 크기(fontsize), 시간의 범위 Xlim, y의 범위 ylim
'''
# raw_all[['count', 'registered', 'casual']].plot(kind = 'line', figsize = (20,6), linewidth= 3, fontsize = 20,
#                                                 xlim = ('2012-01-01', '2012-06-01'), ylim = (0, 1000))
# plt.title('Time Series of Target', fontsize = 20) #시각화 자료의 그림 제목 설정
# plt.xlabel('Index', fontsize = 15)
# plt.ylabel('Demand', fontsize = 15)
# plt.show()

# raw_all[['count']].plot(kind = 'line', figsize = (20, 6), linewidth = 2, fontsize = 20,
#                         xlim = ('2012-01-01', '2012-03-01'), ylim = (0, 1000))
# plt.title('Time Series of Target', fontsize = 20)
# plt.xlabel('Index', fontsize = 15)
# plt.ylabel('Demand', fontsize = 15)
# plt.show()

# split data as trend + seasonal + residual
# 데이터를 추세, 계절성, 잔차로 분리하는 단계
# plt.rcParams['figure.figsize'] = (14, 9) 만약 그림이 깨지면 이것을 이용해서 그림 크기를 조절
# sm.tsa.seasonal_decompose(raw_all['count'], model = 'additive').plot()
# statmodels의 타임시리즈 기법 중에서 이동 평균선을 분해하는 기법으로 model을 additive 혹은 multiplicative 방식으로 그려주십시오
# additive trend + seasonal + residual 이 이렇게 구성되어 있는 것
# plt.show()

result = sm.tsa.seasonal_decompose(raw_all['count'], model = 'additive')
# result.trend # 평균이동선을 토대로 값을 그림으로 나타내기 때문에 처음과 끝 부분이 nan으로 나타남
# result.observed
# result.seasonal
# result.resid
# ((result.observed - result.trend - result.seasonal) == result.resid).sum()
# pd.DataFrame(result.resid).describe()

Y_trend = pd.DataFrame(result.trend)
Y_trend.fillna(method = 'ffill', inplace = True) # 뒤에 있는 nan 값을 채워주는 명령어
Y_trend.fillna(method = 'bfill', inplace = True) # 앞에 있는 nan 값을 채워주는 명령어
Y_trend.columns = ['count_trend']
Y_seasonal = pd.DataFrame(result.seasonal)
Y_seasonal.fillna(method = 'ffill', inplace = True)
Y_seasonal.fillna(method = 'bfill', inplace = True)
Y_seasonal.columns = ['count_seasonal']
# Y_trend.iloc[:20] 데이터의 nan을 확인할 수 있음

# merging several columns
pd.concat([raw_all, Y_trend, Y_seasonal], axis = 1).isnull().sum() # raw_all에다가 Y_trend와 Y_seasonal를 columns(axis)축으로 결합해 주세요
if 'count_trend' not in raw_all.columns:
    if 'count_seasonal' not in raw_all.columns:
        raw_all = pd.concat([raw_all, Y_trend, Y_seasonal], axis = 1)

# raw_all['count'].rolling(24).mean().plot(kind = 'line', figsize = (20,6), linewidth = 3, fontsize = 20,
#                                          xlim = ('2012-01-01', '2012-06-01'), ylim = (0, 1000))
# raw_all에 count 변수에서 앞 뒤로 24개의 데이터를 묶고(rolling) 평균(mean)을 구해서 시각화(plot) 해라

# pd.concat([raw_all['count'],
#            raw_all['count'].rolling(24).mean(),
#            raw_all['count'].rolling(24*7).mean()], axis = 1).plot(kind = 'line', figsize = (20,6), linewidth = 3,
#                                                                   fontsize = 20, xlim = ('2012-01-01', '2013-01-01'),
#                                                                   ylim = (0, 1000))
# plt.title("Time Series of Target", fontsize = 20)
# plt.xlabel("demand", fontsize = 15)
# plt.ylabel('index', fontsize = 15)
# plt.show()

Y_count_Day = raw_all[['count']].rolling(24).mean()
Y_count_Day.fillna(method = 'ffill', inplace = True)
Y_count_Day.fillna(method = 'bfill', inplace = True)
Y_count_Day.columns = ['count_Day']
Y_count_Week = raw_all[['count']].rolling(24*7).mean()
Y_count_Week.fillna(method = 'ffill', inplace = True)
Y_count_Week.fillna(method = 'bfill', inplace = True)
Y_count_Week.columns = ['count_Week']

if 'count_Day' not in raw_all.columns:
    if 'count_Week' not in raw_all.columns:
        raw_all = pd.concat([raw_all, Y_count_Day, Y_count_Week], axis = 1)

# diff() 데이터의 증감폭을 알 수 있다
# raw_all[['count']].diff().plot(kind = 'line', figsize = (20, 6), linewidth = 3, fontsize = 20,
#                                xlim = ('2012-01-01', '2012-06-1'), ylim = (0, 1000))
# plt.xlabel("demand", fontsize = 15)
# plt.ylabel('index', fontsize = 15)
# plt.show()

Y_diff = raw_all[['count']].diff()
Y_diff.fillna(method = 'ffill', inplace = True)
Y_diff.fillna(method = 'bfill', inplace = True)
Y_diff.columns = ['count_diff']

if 'count_diff' not in raw_all.columns:
    raw_all = pd.concat([raw_all, Y_diff], axis = 1)

# print(raw_all[['temp']])
# print(raw_all[['temp']].ndim) # 데이터프레임에 차원에 갯수를 알 수 있음
# pd.cut(raw_all['temp'], 2)[-30:] # 데이터프레임 000의 그룹을 나눈다

raw_all['temp_group'] = pd.cut(raw_all['temp'], 10)
# raw_all.dtypes

# raw_all.describe().T
# raw_all.describe(include = 'all').T # 얘는 문자열까지 전부 표시, 여기서 temp_group은 카테고리이다 보니 전부 nan로 되어 있음음

'''
시간 정보 추출
'''
# raw_all.datetime # raw_all[['datetime']]와 같은 의미
# raw_all.datetime.dt.year # dt 데이터 타임의 프로퍼티 오브젝트이다, 이것은 연도만 추출한다

raw_all['Year'] = raw_all.datetime.dt.year
raw_all['Quarter'] = raw_all.datetime.dt.quarter
# raw_all.describe()
raw_all['Quarter_ver2'] = raw_all['Quarter'] + (raw_all.Year - raw_all.Year.min()) * 4 # dummy variable 활용
raw_all['Month'] = raw_all.datetime.dt.month
raw_all['Day'] = raw_all.datetime.dt.day
raw_all['Hour'] = raw_all.datetime.dt.hour
raw_all['DayofWeek'] = raw_all.datetime.dt.dayofweek

# raw_all.info()
# raw_all.describe(include = 'all').T
#print(raw_all.info())
'''
y = ax라고 해서 y = 매출이고 x = 분기비용이다
분기비용은 2가지 타입으로 나뉘는데 1~4로 나타나는 quarter와 더미 변수를 활용한 1~8로 나뉜 데이터이다

여기서 중요한 것은 어떤 해석을 위해 'quarter'데이터를 사용했는지, 더미 변수를 사용했는지가 중요하다
1. quarter 데이터로 분석시 연도 구분이 없기 때문에 2011년 이든 2012년 이든 1분기에 있었던 계수 'a'의 효과를 표시해 준다
   똑같이 2011년이든 2012년 이든 2분기에 있었던 계수 'a'의 효과를 설명해 준다
   따라서 연도 구분 없이 분기가 바뀌면 보통 어느 정도의 트렌드에 매출 기여도가 있는지를 'a'가 표시해 준다
   
2. 더미 변수를 활용하여 1~8까지 쭉 썼다는 것은 연도 구분을 포함하여 2011~2012년까지 1분기가 증가될 때 마다 
   얼마만큼의 연도의 효과를 매출 기여도에 포함되는지 나타냄
   
결론: 기술적으로 feature를 반영하는것은 어렵지 않다 문제는 기술적으로 전부 집어넣는다고 해서 분석이 쉬워지는 것이 아니라 그게 어떻게 해석이 되고
    사용이 되는지 이해를 하는게 좋다
'''
# raw_all['count'].shift(1) # shift는 특정한 기간만큼 데이터를 옮겨줌, 즉 한 row만큼 이동을 시켜준다

raw_all['count_lag1'] = raw_all['count'].shift(1)
raw_all['count_lag2'] = raw_all['count'].shift(2)

raw_all.count_lag1.fillna(method = 'bfill', inplace = True)
raw_all.count_lag2.fillna(method = 'bfill', inplace = True)
# raw_all.count_lag2.fillna(0) # nan을 int 0으로 전부 치환해라

temp = pd.get_dummies(raw_all['Quarter']) # 데이터프레임 안에 해당 column을 더미 변수를 치환해라
temp2 = pd.get_dummies(raw_all["Quarter"], prefix = "Quarter_Dummy") # 1~4분기 컬럼의 이름이 Quarter_Dummy_1~4로 변경
temp3 = pd.get_dummies(raw_all['Quarter'], prefix = 'Quarter_Dummy', drop_first = True) #첫 번째 있는 column을 삭제함

# 데이터프레임 안에 Quarter라는 columns이 있으면 raw_all 데이터 프레임 뒤에 더미 변수로 Quarter를 변환한 후 결합하고 column을 지워주세요
if 'Quarter' in raw_all.columns:
    raw_all = pd.concat([raw_all, pd.get_dummies(raw_all['Quarter'], prefix = 'Quarter_Dummy', drop_first = True)],
                        axis = 1)
    del raw_all['Quarter']

# [col for col in raw_all.columns if col != 'temp_group']# raw_all 컬럼들을 반복문으로 돌리는데 temp_group이 아닌것들만 모아서 리스트로 만들어 주세요

def datasplit_cs(raw, Y_colname, X_colname, test_size, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(raw[X_colname], raw[Y_colname], test_size = test_size,
                                                        random_state = random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test

def datasplit_ts(raw, Y_colname, X_colname, test_size, random_state, criteria):
    raw_train = raw.loc[raw.index < criteria,:]
    raw_test = raw.loc[raw.index >= criteria,:]

    X_train = raw_train[X_colname]
    Y_train = raw_train[Y_colname]
    X_test = raw_test[X_colname]
    Y_test = raw_test[Y_colname]

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print(X_train.shape)
    return X_train, Y_train, X_test, Y_test

def evaluation(Y_real, Y_pred, graph_on):
    loss_length = len(Y_real.values.flatten())-len(Y_pred)
    if loss_length != 0:
        Y_real = Y_real[graph_on:]
    if graph_on == True:
        pd.concat([Y_real, pd.DataFrame(Y_pred, index=Y_pred.index, columns=['Prediction'])], axis=1).plot(kind='line', figsize=(20,6),
                                                                                                           xlim=(Y_real.index.min(), Y_real.index.max()),
                                                                                                           fontsize=15, linewidth=3)
        plt.title('Time Series of Target', fontsize=20)
        plt.xlabel('index', fontsize=15)
        plt.ylabel('demand', fontsize=15)

    MAE = abs(Y_real.values.flatten()-Y_pred).mean()
    MSE = ((Y_real.values.flatten()-Y_pred)**2).mean()
    MAPE = abs((Y_real.values.flatten()-Y_pred)/Y_real.values.flatten()*100).mean()

    Score = pd.DataFrame([MAE, MSE, MAPE], index=['MAE', 'MSE', 'MAPE'], columns=['Score'])
    Residual = pd.DataFrame(Y_real.values.flatten()-Y_pred, index=Y_real.index, columns=['Error'])

    return Score, Residual
