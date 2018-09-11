import keras as kr
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import holidays
import calendar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,Normalizer,MinMaxScaler,QuantileTransformer
from datetime import datetime
from sklearn.model_selection import train_test_split,learning_curve, ShuffleSplit,cross_val_score
from sklearn.metrics import r2_score, mean_squared_error,make_scorer
from keras.layers import Input,Dense,LSTM,Dropout,TimeDistributed,ConvLSTM2D,Conv3D,Reshape,BatchNormalization,Flatten,Embedding,LeakyReLU,ELU,ThresholdedReLU,PReLU,GRU,Highway
from keras.layers.noise import GaussianNoise,GaussianDropout
from keras.activations import tanh
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
K.set_image_dim_ordering('th')


datapath = 'E:\\OneDrive - Georgia State University\\ML Project\\Amusement Park\\Data'
# datapath = '/Users/ace/Desktop/OneDrive - Georgia State University/ML Project/Amusement Park/Data'
os.chdir(datapath)

data = pd.read_csv('Train.csv')
# final = pd.read_csv('finished.csv', index_col=0)
def clean(df, n =24):
    s = df['TimeStamp'].apply(lambda i: i.split(' '))
    df.loc[:,'Date'] = s.apply(lambda i: i[0])
    df.loc[:,'Time'] = s.apply(lambda i: i[1])
    usholiday = holidays.UnitedStates()
    df.loc[:,'TimeStamp'] = pd.to_datetime(df['TimeStamp'],format= '%Y-%m-%d %H:%M:%S',infer_datetime_format=True)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d',infer_datetime_format=True)
    df.loc[:,'Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df.loc[:,'Date'] = df['Date'].map(lambda x: x.date())
    df.loc[:,'Time'] = df['Time'].map(lambda x: x.time())
    df.loc[:,'Weekday(num)'] = df['Date'].map(lambda x: x.weekday())
    df.loc[:,'Hour'] = df['Time'].map(lambda x: x.hour)
    df.loc[:,'Year'] = df['Date'].map(lambda x: x.year)
    df.loc[:,'Month'] = df['Date'].map(lambda x: x.month)
    df.loc[:,'Day'] = df['Date'].map(lambda x: x.day)
    df.loc[:,'Holiday'] = df['Date'].map(lambda x: 1 if x in usholiday else 0)
    df = df.drop(['Date','Time'],axis=1)
    mm = MinMaxScaler((0,1))
    df['Humidity'] = mm.fit_transform(df['Humidity'].values.reshape(-1,1))
    df = df.sort_values(['TimeStamp'])
    df = df.set_index('TimeStamp')
    df = df.reset_index(drop=True)
    df1 = df.drop(['Ticket1','Ticket2'],axis=1)
    i = 1
    while i <n:
        df2 = pd.DataFrame()
        df2 = df1.shift(i)
        name = '- %s hour'%i
        df2 = df2.add_suffix(name)
        df = df.join(df2)
        i+=1
    df = df.dropna(axis=0)
    df.to_csv('LTSM.csv')
    return df


def LTSM(df,step=1):
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    y = df[['Ticket1','Ticket2']].as_matrix()
    x = df.drop(['Ticket1','Ticket2'],axis=1).as_matrix()
    # x= MinMaxScaler((0,1)).fit_transform(x)
    x = PCA().fit_transform(x)
    # x = PolynomialFeatures(2).fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x_train.reshape((x_train.shape[0],step,int(x_train.shape[1]/step)))
    x_test = x_test.reshape((x_test.shape[0],step,int(x_test.shape[1]/step)))
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    ES = EarlyStopping( monitor='loss',patience=5000,verbose=1,mode='min')
    RS = ReduceLROnPlateau(monitor='loss',patience=50,verbose=1,factor=0.5)
    model = kr.Sequential()
    model.add(BatchNormalization(input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(GaussianNoise(0.5))
    model.add(LSTM(24 , return_sequences=True,kernel_initializer='he_normal',use_bias=True,bias_initializer=kr.initializers.one(),unit_forget_bias=True,kernel_regularizer=kr.regularizers.l1_l2(0.001,0.0001)))
    model.add(LeakyReLU())
    model.add(LSTM(24, return_sequences=False, go_backwards=True,kernel_initializer='he_normal'))
    # model.add(LSTM(16, return_sequences=False, go_backwards=True,kernel_initializer='he_normal'))
    model.add(GaussianDropout(0.5))
    # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(Dense(32))
    # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(Highway())
    # model.add(Dense(64))
    # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(Highway())
    # model.add(Dense(128))
    # model.add(LeakyReLU())
    # model.add(BatchNormalization())
    # model.add(Highway())
    # model.add(Dense(256))
    model.add(LeakyReLU())
    # model.add(BatchNormalization())
    model.add(Dense(2))
    sgd = kr.optimizers.sgd(lr=0.1, momentum=0.01,decay=0.001,nesterov=True,clipnorm=3)
    model.compile(loss='mape', optimizer=sgd, metrics=['mae', 'mse'])
    his = model.fit(x=x_train,y=y_train,epochs=50000,batch_size=3000,validation_split=0.3,callbacks=[RS,ES],shuffle=True)
    y_pre = model.predict(x_test,batch_size=3000)
    plt.plot(his.history['loss'],label='train')
    plt.plot(his.history['val_loss'],label='test')
    plt.title('LSTM Performance')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()
    scores = model.evaluate(x_test, y_test, verbose=0,batch_size=3000)
    r2 = r2_score(y_test,y_pre)
    mse = mean_squared_error(y_test,y_pre)
    print(scores)
    print("LTSM MAPE: %.2f%%" % (scores[2]))
    print('R2 Score is ',r2)
    print('MSE is ',mse)
    return y_pre


def lstm():
    model = kr.Sequential()
    model.add(BatchNormalization(input_shape=(1, 465)))
    model.add(LSTM(64, return_sequences=True, kernel_initializer='he_normal', use_bias=True,
                   bias_initializer=kr.initializers.one(), unit_forget_bias=True,
                   kernel_regularizer=kr.regularizers.l1_l2(0.001, 0.0001)))
    model.add(LeakyReLU())
    model.add(LSTM(64, return_sequences=False, go_backwards=True, kernel_initializer='he_normal'))
    model.add(Highway())
    model.add(GaussianDropout(0.5))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(32))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Highway())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Highway())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Highway())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dense(1))
    sgd = kr.optimizers.sgd(lr=0.1, momentum=0.1, decay=0.001, nesterov=True, clipnorm=3)
    model.compile(loss='mape', optimizer=sgd, metrics=['mae', 'mse'])
    return model

def cross_val(df,step=1):
    y = df[['duration']].as_matrix().reshape(-1,1)
    x = df.drop('duration',axis=1).as_matrix()
    x= MinMaxScaler((0,1)).fit_transform(x)
    x = PCA().fit_transform(x)
    x = PolynomialFeatures(2).fit_transform(x)
    x = x.reshape((x.shape[0],step,int(x.shape[1]/step)))
    ES = EarlyStopping(monitor='loss', patience=200, verbose=1, mode='min')
    RS = ReduceLROnPlateau(monitor='loss', patience=20, verbose=1, factor=0.5)
    lmodel = KerasRegressor(build_fn=lstm,epochs=10000,batch_size=30000,verbose=1)
    a = cross_val_score(lmodel,x,y,cv=10,fit_params={'callbacks':[RS,ES]},scoring=make_scorer(r2_score,greater_is_better=True))
    return a


