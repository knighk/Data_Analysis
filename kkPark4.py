import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import holidays
from sklearn import preprocessing

def changeTimeStamp(df):
    usholiday = holidays.UnitedStates()
    df['TimeStampTemp']=df["TimeStamp"].copy()
    df['Date'] = pd.to_datetime(df['TimeStampTemp'].apply(lambda i: i.split(' ')[0])).map(lambda x: x.date())
    df['Year'] = df['Date'].map(lambda x: x.year)
    df['Month'] = df['Date'].map(lambda x: x.month)
    #df['Day'] = df['TimeStampTemp'].apply(lambda i: i.split(' ')[0].split('-')[-1])
    df['Hour'] = pd.to_datetime(df['TimeStampTemp'].apply(lambda i: i.split(' ')[1])).map(lambda x: x.time().hour)
    df['Weekday'] = df['Date'].map(lambda x: x.weekday())
    df['Holiday'] = df['Date'].map(lambda x: 1 if x in usholiday else 0)
    df.reset_index()
    return df

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

# def outliers_iqr(ys):
#     quartile_1, quartile_3 = np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((ys > upper_bound) | (ys < lower_bound))

pd.set_option('display.max_columns', None)
path = "F:\\MSA\\machine learning\\project\\Amusement Park\\kk\\"
df = pd.read_csv(path + "Train.csv").sort_values("TimeStamp",ascending=0)
# print df.dtypes
df["Humidity"] = preprocessing.scale(df["Humidity"])
df=changeTimeStamp(df)
df = df.drop(['TimeStamp',"TimeStampTemp","Date","Holiday"], axis=1)
# pyplot.figure(figsize=(20, 6))
# pyplot.plot( df["Ticket1"].values)
# pyplot.show()
#
# pyplot.figure(figsize=(20, 6))
# pyplot.plot(df["Ticket1"].values[:50])
# pyplot.show()

window_size = 2
dfTemp = df.copy().drop(["Ticket1","Ticket2","Year","Month","Hour","Weekday","Wind"], axis=1)
for i in range(window_size):
    df = pd.concat([df, dfTemp.shift(-(i + 1)).add_suffix("_t_"+str(i+1))], axis=1)

df.dropna(axis=0, inplace=True)

# df=df.set_index(df["TimeStamp"])
df.to_csv('F:\MSA\machine learning\project\Amusement Park\kk\df4.csv',index=False)

#75
# Model Performance
# Average Error: 17.5102 degrees.
# Accuracy = 48.12%.

#48
# Model Performance
# Average Error: 17.2632 degrees.
# Accuracy = 47.06%.

#24
# Model Performance
# Average Error: 16.5764 degrees.
# Accuracy = 50.62%.

#8
# Model Performance
# Average Error: 16.5862 degrees.
# Accuracy = 52.63%.

#6
# Model Performance
# Average Error: 16.5675 degrees.
# Accuracy = 52.79%.

#3
# Model Performance
# Average Error: 16.6738 degrees.
# Accuracy = 53.73%.

#1
# Model Performance
# Average Error: 16.5660 degrees.
# Accuracy = 52.80%.