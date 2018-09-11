"""##Packages"""
import os, sys, stat
import pandas as pd
import datetime as dt
import numpy as np
import math
import seaborn as sns
import calendar
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
from sklearn import cluster
from sklearn import mixture
from sklearn import metrics
from scipy.spatial import distance
import warnings
from sklearn.model_selection import ParameterGrid
import gc
import itertools
import holidays
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import gpxpy.geo
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,IsolationForest
from sklearn import metrics
from sklearn.model_selection import learning_curve, ShuffleSplit
import multiprocessing as mp
from sklearn.feature_selection import RFECV, SelectKBest,mutual_info_regression

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


n_jobs=70
# datapath = '/Users/ace/Desktop/OneDrive - Georgia State University/ML Project/Gotham Cabs/Data'
datapath = 'E:\\OneDrive - Georgia State University\\ML Project\\Gotham Cabs\\Data'
# datapath = '/home/ubuntu/gt'

os.chdir(datapath)
test = pd.read_csv('final1_test.csv', engine='python')
train = pd.read_csv('final1_train.csv',engine='python')

# tt = pd.read_csv('cd1.csv',index_col=0)
# final = pd.read_csv('final1.csv', index_col=0)


# '##Final Visualization'
def box_plot(df):
    blist = ['#e7eeff', '#a188ff', '#6c88cc', '#44557f','#50447f']
    bmain = sns.color_palette(blist)
    x = df.drop('pickup_datetime',axis=1)
    x[['duration', 'pickup_x', 'pickup_y', 'dropoff_x','dropoff_y']] = StandardScaler().fit_transform(x.as_matrix())
    x1 = lof_completed(x)
    plt.figure(figsize=(36,18))
    plt.subplot(121)
    sns.boxplot(data = x,palette=bmain)
    sns.despine()
    plt.title('Before')
    plt.xlabel('Columns')
    plt.ylabel('Standarded Value')
    plt.subplot(122)
    sns.boxplot(data = x1,palette=bmain)
    sns.despine()
    plt.title('After')
    plt.xlabel('Columns')
    plt.ylabel('Standarded Value')
    plt.show()


def importance(df):
    df['RF ranking'] = 1/df['RF ranking']
    df['EDT ranking'] = 1/df['EDT ranking']
    df['XGboost ranking'] = 1/df['XGboost ranking']
    df['GBDT ranking'] = 1/df['GBDT ranking']
    plt.figure(figsize=(24,24))
    sns.barplot(x = df.index,y = 'RF ranking',data=df)
    plt.title('Feature Importance for Random Forest',y=1)
    sns.despine()
    plt.show()
    plt.figure(figsize=(24,24))
    sns.barplot(x = df.index,y = 'EDT ranking',data=df)
    plt.title('Feature Importance for Extra Decision Tree',y=1)
    sns.despine()
    plt.show()
    plt.figure(figsize=(24,24))
    sns.barplot(x = df.index,y = 'XGboost ranking',data=df)
    plt.title('Feature Importance for XGboost',y=1)
    sns.despine()
    plt.show()
    plt.figure(figsize=(24,24))
    sns.barplot(x = df.index,y = 'GBDT ranking',data=df)
    plt.title('Feature Importance for GBDT',y=1)
    sns.despine()
    plt.show()


# "##Clean"
def clean(df):
    pd.options.mode.chained_assignment = None
    usholiday = holidays.UnitedStates()
    print('start')
    s = df['pickup_datetime'].apply(lambda i: i.split(' '))
    df['Date'] = s.apply(lambda i: i[0])
    df['Time'] = s.apply(lambda i: i[1])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format= '%Y-%m-%d %H:%M:%S',infer_datetime_format=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d',infer_datetime_format=True)
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Date'] = df['Date'].map(lambda x: x.date())
    nearholiday1 = list(map(lambda x: x + dt.timedelta(days=1),list(usholiday.keys())))
    nearholiday2 = list(map(lambda x: x - dt.timedelta(days=1),list(usholiday.keys())))
    df['Near Holiday'] = (df['Date'].isin(nearholiday1) | df['Date'].isin(nearholiday2))
    df['Near Holiday'] = df['Near Holiday'].map(lambda x: 1 if x == True else 0)
    df['Day of Year'] = df['Date'].map(lambda x: x.timetuple().tm_yday)
    df['Day of Year'] = cluster.KMeans(n_clusters=24).fit_predict(df['Day of Year'].values.reshape(-1,1))
    df['Week of Year'] = df['Date'].map(lambda x: x.isocalendar()[1])
    df['Week of Year'] = cluster.KMeans(n_clusters=24).fit_predict(df['Week of Year'].values.reshape(-1,1))
    df['Time'] = df['Time'].map(lambda x: x.time())
    df['Weekday(num)'] = df['Date'].map(lambda x: x.weekday())
    df['Hour'] = df['Time'].map(lambda x: x.hour)
    df['Minutes'] = df['Time'].map(lambda x: x.minute)
    df['Pick Up Minute of the Day'] = df['Minutes'] + df['Hour']*60
    df['Pick Up Minute of the Day'] = cluster.KMeans(n_clusters=24).fit_predict(df['Pick Up Minute of the Day'].values.reshape(-1,1))
    df['Year'] = df['Month'] = df['Date'].map(lambda x: x.year)
    df['Month'] = df['Date'].map(lambda x: x.month)
    df['Day'] = df['Date'].map(lambda x: x.day)
    df['Weekday'] = df['Date'].map(lambda x: calendar.day_name[x.weekday()])
    df['Pick up is Weekend'] = df['Weekday(num)'].map(lambda x: 1 if x >= 5 else 0)
    df['Holiday'] = df['Date'].map(lambda x: 1 if x in usholiday else 0)
    rides = pd.DataFrame(df.groupby(['Day of Year', 'Week of Year', 'Weekday(num)', 'Year', 'Month', 'Day', 'Weekday','Pick up is Weekend', 'Holiday']).size()).reset_index()
    rides['rides'] = rides[0]
    rides = rides.drop(0,1)
    df = df.merge(rides, on=['Day of Year', 'Week of Year', 'Weekday(num)', 'Year', 'Month', 'Day', 'Weekday','Pick up is Weekend', 'Holiday'],how='left')
    print('complete rides start distance')
    # df['braycurtis distance'] = df.index.map(lambda x: metrics.pairwise_distances(df.loc[x, ['pickup_x', 'pickup_y']].reshape(1,-1),df.loc[x, ['dropoff_x', 'dropoff_y']].reshape(1,-1), metric='braycurtis',n_jobs=n_jobs)[0,0])
    df['euclidean distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup_x','pickup_y']].reshape(1,-1),df.loc[x,['dropoff_x','dropoff_y']].reshape(1,-1),metric='euclidean')[0,0])
    print('compelte euclidean')
    df['manhattan distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup_x','pickup_y']].reshape(1,-1),df.loc[x,['dropoff_x','dropoff_y']].reshape(1,-1),metric='manhattan')[0,0])
    print('compelte manhattan')
    df['cosine distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup_x','pickup_y']].reshape(1,-1),df.loc[x,['dropoff_x','dropoff_y']].reshape(1,-1),metric='cosine')[0,0])
    print('compelte cosine')
    pca = PCA(n_components=1)
    df['pickup_pca'] = pca.fit_transform(df[['pickup_x', 'pickup_y']])
    df['dropoff_pca'] = pca.fit_transform(df[['dropoff_x', 'dropoff_y']])
    df = df.drop(['pickup_datetime','Weekday','Date','Time'],axis=1)
    return df


# '##Plot Gotham！！！！！'
def plot(df,test):
    df1 = split0(df)
    df2 = split0(test)
    cm = sns.diverging_palette(240,10,as_cmap=True)
    plt.figure(figsize=(12,12))
    plt.scatter(x = df1['x'], y = df1['y'],cmap=cm)
    plt.scatter(x = df2['x'],y=df2['y'],c='r')
    plt.show()

# '##除噪'
def out(df):
    l = ['pickup_y','dropoff_y']
    for x in l:
        df = df[df[x]>=df[x].quantile(0.01)]
        df = df[df[x]<=df[x].quantile(0.99)]
    return df
# '##Used on lstm4.csv'

def lof_completed(df):
    with pd.option_context('mode.use_inf_as_null', True):
        df = df.dropna()
    x = df[['pickup_x','pickup_y','dropoff_x','dropoff_y']].as_matrix()
    model = LocalOutlierFactor(contamination=0.35,n_jobs=n_jobs,algorithm='kd_tree',leaf_size=50,n_neighbors=50)
    x_pre = model.fit_predict(x)
    df['outlier'] = x_pre
    new_df = df[df['outlier'] == 1]
    del new_df['outlier']
    new_df = new_df.reset_index(drop=True)
    # plot(df1)
    return new_df



def iforest(df):
    with pd.option_context('mode.use_inf_as_null', True):
        df = df.dropna()
    y = df['duration'].as_matrix()
    x = df[['braycurtis distance','canberra distance']].as_matrix()
    model = IsolationForest(bootstrap=True,n_estimators=200)
    model.fit(x,y)
    x_pre = model.predict(x)
    df['outlier'] = x_pre
    new_df = df[df['outlier'] == 1]
    del new_df['outlier']
    new_df = new_df.reset_index(drop=True)
    # plot(df1)
    return new_df


def pca(df):
    x1 = df[['pickup_x','pickup_y']].as_matrix()
    pick_pca = PCA()
    pick_pca.fit(x1)
    x2 = df[['dropoff_x','dropoff_y']].as_matrix()
    drop_pca = PCA()
    drop_pca.fit(x2)
    # plt.figure(figsize=(24,12))
    # plt.subplot(121)
    # plt.plot(pick_pca.explained_variance_,linewidth=2)
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    # plt.title('Pick Up')
    # plt.subplot(122)
    # plt.plot(drop_pca.explained_variance_,linewidth=2)
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    # plt.title('Drop Off')
    # plt.show()



# '##Clustering'
# df_all = pd.read_csv('xy.csv',index_col=0)
def clustering1(df,n=4):
    df = split0(df)
    x = df[['x','y']].as_matrix()
    model = cluster.MiniBatchKMeans(n_clusters=n)
    y_pre = model.fit_predict(x)
    chs = metrics.calinski_harabaz_score(x, y_pre)
    print('The score for MiniBatchKMeans is',chs)
    plt.figure(figsize=(24,12))
    plt.subplot(121)
    plt.title('MiniBatchKMeans')
    plt.scatter(x[:,0],x[:,1], c= y_pre)
    model1 = cluster.KMeans(n_clusters=n)
    model1.fit(x)
    y_pre_1 = model1.predict(x)
    chs1 = metrics.calinski_harabaz_score(x, y_pre_1)
    print('The score for Kmeans is', chs1)
    plt.subplot(122)
    plt.title('Kmeans Clustering')
    plt.scatter(x[:,0],x[:,1], c= y_pre_1)
    sns.despine()
    plt.show()


def clustering2(df,n=5):
    x = df[['x','y']].as_matrix()
    y = df['speed'].reshape(1,-1)
    model = cluster.MiniBatchKMeans(n_clusters=n)
    y_pre = model.fit_predict(x, y)
    chs = metrics.calinski_harabaz_score(x, y_pre)
    print('The score for MiniBatchKMeans is',chs)
    # plt.figure(figsize=(36,12))
    # plt.subplot(131)
    # plt.title('MiniBatchKMeans')
    # plt.scatter(x[:,0],x[:,1], c= y_pre)
    # model1 = cluster.KMeans(n_clusters=n)
    # y_pre_1 = model1.fit_predict(x, y)
    # chs1 = metrics.calinski_harabaz_score(x, y_pre_1)
    # print('The score for KMeans is', chs1)
    # plt.subplot(132)
    # plt.title('KMeans Clustering')
    # plt.scatter(x[:,0],x[:,1], c= y_pre_1)
    # model2 = cluster.Birch(n_clusters=n)
    # y_pre_2 = model2.fit_predict(x, y)
    # chs2 = metrics.calinski_harabaz_score(x, y_pre_2)
    # print('The score for Birch is', chs2)
    # plt.subplot(133)
    # plt.title('Birch Clustering')
    # plt.scatter(x[:,0],x[:,1], c= y_pre_2)
    # plt.show()


def split0(train,test):
    train1 = train[['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    train2 = train[['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    test1 = test[['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    test2 = test[['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([train1,train2,test1,test2])
    return df3



def split(train,test,i):
    train1 = train[train['pickup 1']==i][['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    train2 = train[train['dropoff 1']==i][['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    test1 = test[test['pickup 1']==i][['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    test2 = test[test['dropoff 1']==i][['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([train1,train2,test1,test2])
    return df3


def split2(train,test,i,b):
    train1 = train[(train['pickup 1']==i) & (train['pickup 2'] == b)][['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    train2 = train[(train['dropoff 1']==i) & (train['dropoff 2'] ==b)][['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    test1 = test[(test['pickup 1']==i) & (test['pickup 2'] == b)][['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    test2 = test[(test['dropoff 1']==i) & (test['dropoff 2'] ==b)][['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([train1,train2,test1,test2])
    return df3
# '##划分中区'


def second(train,test,n=8):
    clist = list(set(train['pickup 1']))
    ps_train = pd.DataFrame()
    ds_train = pd.DataFrame()
    ps_test = pd.DataFrame()
    ds_test = pd.DataFrame()
    for i in clist:
        pick_test=pd.DataFrame()
        pick_train=pd.DataFrame()
        drop_test=pd.DataFrame()
        drop_train=pd.DataFrame()
        df1 = split(train,test,i)
        print('complete split for ',i)
        x = df1[['x','y']].as_matrix()
        model = cluster.Birch(n_clusters=n)
        print('complete model for',i)
        model.fit(x)
        print('complete fit')
        name = 'Second Clustering for Clustering %s' %i
        pick_train = train[train['pickup 1']==i][['pickup 1','pickup_x','pickup_y']]
        drop_train = train[train['dropoff 1']==i][['dropoff 1','dropoff_x','dropoff_y']]
        pick_test = test[test['pickup 1']==i][['pickup 1','pickup_x','pickup_y']]
        drop_test = test[test['dropoff 1']==i][['dropoff 1','dropoff_x','dropoff_y']]
        pick_train['pickup 2'] = pick_train.index.map(lambda x: int(model.predict(train.loc[x, ['pickup_x','pickup_y']].as_matrix().reshape(1,-1))))
        drop_train['dropoff 2'] = drop_train.index.map(lambda x: int(model.predict(train.loc[x, ['dropoff_x','dropoff_y']].as_matrix().reshape(1,-1))))
        pick_test['pickup 2'] = pick_test.index.map(lambda x: int(model.predict(test.loc[x, ['pickup_x','pickup_y']].as_matrix().reshape(1,-1))))
        drop_test['dropoff 2'] = drop_test.index.map(lambda x: int(model.predict(test.loc[x, ['dropoff_x','dropoff_y']].as_matrix().reshape(1,-1))))
        ps_train = pd.concat([ps_train,pick_train])
        ds_train = pd.concat([ds_train,drop_train])
        ps_test = pd.concat([ps_test,pick_test])
        ds_test = pd.concat([ds_test,drop_test])
        print('compelte label',i)
    ps_train = ps_train.drop(['pickup 1','pickup_x','pickup_y'],axis=1)
    ds_train = ds_train.drop(['dropoff 1','dropoff_x','dropoff_y'],axis=1)
    ps_test = ps_test.drop(['pickup 1','pickup_x','pickup_y'],axis=1)
    ds_test = ds_test.drop(['dropoff 1','dropoff_x','dropoff_y'],axis=1)
    train = train.merge(ps_train,left_index=True,right_index=True,copy=False)
    train = train.merge(ds_train,left_index=True,right_index=True,copy=False)
    test = test.merge(ps_test,left_index=True,right_index=True,copy=False)
    test = test.merge(ds_test,left_index=True,right_index=True,copy=False)
    return train,test





def third(train,test,n=8):
    clist = list(set(train['pickup 1']))
    vlist = list(set(train['pickup 2']))
    ps_train = pd.DataFrame()
    ds_train = pd.DataFrame()
    ps_test = pd.DataFrame()
    ds_test = pd.DataFrame()
    nlist = list(itertools.product(clist, vlist))
    for i,b in nlist:
        pick_test=pd.DataFrame()
        pick_train=pd.DataFrame()
        drop_test=pd.DataFrame()
        drop_train=pd.DataFrame()
        df1 = split2(train,test,i,b)
        print('complete split for ',i,b)
        x = df1[['x','y']].as_matrix()
        model = cluster.Birch(n_clusters=n)
        print('complete model for',i,b)
        model.fit(x)
        print('complete fit')
        pick_train = train[(train['pickup 1']==i)&(train['pickup 2'] == b)][['pickup 1','pickup 2','pickup_x', 'pickup_y']]
        drop_train = train[(train['dropoff 1']==i)&(train['dropoff 2'] == b)][['dropoff 1','dropoff 2','dropoff_x', 'dropoff_y']]
        pick_test = test[(test['pickup 1']==i)&(test['pickup 2'] == b)][['pickup 1','pickup 2','pickup_x', 'pickup_y']]
        drop_test = test[(test['dropoff 1']==i)&(test['dropoff 2'] == b)][['dropoff 1','dropoff 2','dropoff_x', 'dropoff_y']]
        pick_train['pickup 3'] = pick_train.index.map(lambda x: int(model.predict(train.loc[x, ['pickup_x', 'pickup_y']].as_matrix().reshape(1, -1))))
        drop_train['dropoff 3'] = drop_train.index.map(lambda x: int(model.predict(train.loc[x, ['dropoff_x', 'dropoff_y']].as_matrix().reshape(1, -1))))
        pick_test['pickup 3'] = pick_test.index.map(lambda x: int(model.predict(test.loc[x, ['pickup_x', 'pickup_y']].as_matrix().reshape(1, -1))))
        drop_test['dropoff 3'] = drop_test.index.map(lambda x: int(model.predict(test.loc[x, ['dropoff_x', 'dropoff_y']].as_matrix().reshape(1, -1))))
        ps_train = pd.concat([ps_train,pick_train])
        ds_train = pd.concat([ds_train,drop_train])
        ps_test = pd.concat([ps_test,pick_test])
        ds_test = pd.concat([ds_test,drop_test])
        print('compelte first label',i,' second label',b)
    ps_train = ps_train.drop(['pickup 1','pickup 2','pickup_x', 'pickup_y'],axis=1)
    ds_train = ds_train.drop(['dropoff 1','dropoff 2','dropoff_x', 'dropoff_y'],axis=1)
    ps_test = ps_test.drop(['pickup 1','pickup 2','pickup_x', 'pickup_y'],axis=1)
    ds_test = ds_test.drop(['dropoff 1','dropoff 2','dropoff_x', 'dropoff_y'],axis=1)
    train = train.merge(ps_train,left_index=True,right_index=True,copy=False)
    train = train.merge(ds_train,left_index=True,right_index=True,copy=False)
    test = test.merge(ps_test,left_index=True,right_index=True,copy=False)
    test = test.merge(ds_test,left_index=True,right_index=True,copy=False)
    return train,test


# '##Choose Birch for clustering 2'
def gr_cv(df):
    x = df[['x','y']].as_matrix()
    nlist = list(range(3,61))
    l2 = []
    for n in nlist:
        model = mixture.GaussianMixture(n_components=n)
        model.fit(x)
        y_pre = model.predict(x)
        plt.figure(figsize=(12,12))
        name = 'GR %s clusters'% n
        plt.title(name)
        plt.scatter(x[:, 0], x[:, 1], c=y_pre)
        plt.show()
        chs = metrics.calinski_harabaz_score(x,y_pre)
        l2.append((chs, n))
        print('Score for this fit is',chs)
    return max(l2)


# '##Choose mini for clustering 1'


def kmeans_cv(df):
    df1 = df[['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    df2 = df[['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([df1,df2])
    x = df3[['x','y']].as_matrix()
    nlist = list(range(10,61))
    hyperparams = {'n_clusters':nlist,'init': ['k-means++']}
    l1 = list(ParameterGrid(hyperparams))
    l2 = []
    for i in l1:
        gc.enable()
        gc.collect()
        model = cluster.KMeans(**i)
        y_pre = model.fit_predict(x)
        name = str(i)
        # plt.figure(figsize=(12,12))
        # plt.title(name)
        # plt.scatter(x[:, 0], x[:, 1], c=y_pre)
        # plt.show()
        chs = metrics.calinski_harabaz_score(x,y_pre)
        l2.append((chs, i))
        print('Score for this fit is',chs)
    return max(l2)


def kmeans_finished(df):
    df1 = df[['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    df2 = df[['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([df1,df2])
    x = df3[['x','y']].as_matrix()
    gc.enable()
    gc.collect()
    model = cluster.KMeans(n_clusters=58)
    y_pre = model.fit_predict(x)
    l1 = [['pickup_x','pickup_y'],['dropoff_x','dropoff_y']]
    # plt.figure(figsize=(12,12))
    # plt.scatter(x[:, 0], x[:, 1], c=y_pre)
    # plt.show()
    for i in l1:
       x_pre = model.predict(df[i].as_matrix())
       name = 'pickup 1' if l1.index(i) == 0 else 'dropoff 1'
       # plt.scatter(df[i][0], df[i][1], c=x_pre)
       # plt.show
       df[name] = x_pre
    dummies = ['pickup 1','dropoff 1']
    for feature in dummies:
        dummy_features = pd.get_dummies(df[feature], prefix=feature)
        for dummy in dummy_features:
            df[dummy] = dummy_features[dummy]
        df = df.drop([feature], 1)
    new_df = df
    return new_df


def mini_cv(df):
    df1 = df[['pickup_x','pickup_y']].rename(columns={'pickup_x':'x','pickup_y':'y'})
    df2 = df[['dropoff_x','dropoff_y']].rename(columns={'dropoff_x':'x','dropoff_y':'y'})
    df3 = pd.concat([df1,df2])
    x = df3[['x','y']].as_matrix()
    nlist = list(range(3,61))
    hyperparams = {'n_clusters': nlist,'init': ['k-means++', 'random'], 'batch_size': [100, 200, 300, 400, 500,600,700,800,900,1000]}
    l1 = list(ParameterGrid(hyperparams))
    l2 = []
    for i in l1:
        gc.enable()
        gc.collect()
        model = cluster.MiniBatchKMeans(**i)
        y_pre = model.fit_predict(x)
        name = str(i)
        # plt.figure(figsize=(12,12))
        # plt.title(name)
        # plt.scatter(x[:, 0], x[:, 1], c=y_pre)
        # plt.show()
        chs = metrics.calinski_harabaz_score(x,y_pre)
        l2.append((chs, i))
        print('Score for this fit is',chs)
    return max(l2)


def mini_finished(train,test,n=6):
    x = split0(train,test)
    model = cluster.MiniBatchKMeans(n_clusters=n)
    l1 = [['pickup_x','pickup_y'],['dropoff_x','dropoff_y']]
    model.fit(x)
    l = model.cluster_centers_
    for i in l1:
       train_pre = model.predict(train[i].as_matrix())
       test_pre = model.predict(test[i].as_matrix())
       name = 'pickup 1' if l1.index(i) == 0 else 'dropoff 1'
       namex = 'pickup 1 center x' if l1.index(i) == 0 else 'dropoff 1 center x'
       namey = 'pickup 1 center y' if l1.index(i) == 0 else 'dropoff 1 center y'
       train[name] = train_pre
       test[name] = test_pre
       for x in list(range(n)):
           train.loc[train[name] == x, namex] = l[x,0]
           train.loc[train[name] == x, namey] = l[x,1]
           test.loc[test[name] == x, namex] = l[x,0]
           test.loc[test[name] == x, namey] = l[x,1]
    return train,test


def dummy(df):
    dummies = ['Near Holiday', 'Day of Year', 'Week of Year', 'Weekday(num)',
       'Hour', 'Minutes', 'Pick Up Minute of the Day', 'Year', 'Month',
       'Day', 'Pick up is Weekend', 'Holiday', 'pickup 1', 'dropoff 1','dropoff 2', 'pickup 2', 'dropoff 3', 'pickup 3', 'Minutes_bins',]
    for feature in dummies:
        dummy_features = pd.get_dummies(df[feature], prefix=feature)
        for dummy in dummy_features:
            df[dummy] = dummy_features[dummy]
        df = df.drop([feature], 1)
    return df


def group(df,test):
    plist = ['pickup 1', 'dropoff 1','pickup 2', 'dropoff 2','pickup 3', 'dropoff 3']
    tlist = ['Year','Month','Day', 'Hour']
    for x in plist:
        clist = [x]
        for i in tlist:
            clist.append(i)
            print(clist)
            name = str(x)+' '+str(i)
            rides = pd.DataFrame(df.groupby(clist).size()).reset_index()
            rides[name] = rides[0]
            rides = rides.drop(0, 1)
            df = df.merge(rides, on=clist, how='left')
            test = test.merge(rides,on=clist,how='left')
    return df,test


def group_duration(df,test):
    plist = [['pickup 1', 'dropoff 1'],['pickup 1', 'dropoff 1','pickup 2', 'dropoff 2'],['pickup 1', 'dropoff 1','pickup 2', 'dropoff 2', 'pickup 3', 'dropoff 3']]
    tlist = ['Year','Month','Day']
    onelist = ['pickup 1', 'dropoff 1']
    for x in plist:
        clist = x
        name_mean = 'clustering mean duration ' + str(plist.index(x) + 1)
        dur = pd.DataFrame(df.groupby(x)['duration'].mean()).rename(columns={'duration': name_mean}).reset_index()
        df = df.merge(dur, on=x, how='left')
        test = test.merge(dur, on=x, how='left')
    for i in tlist:
        onelist.append(i)
        print(onelist)
        name_mean = str(i) + ' clustering mean duration ' + str(1)
        dur1 = pd.DataFrame(df.groupby(onelist)['duration'].mean()).rename(columns={'duration':name_mean}).reset_index()
        df = df.merge(dur1, on=onelist, how='left')
        test = test.merge(dur1, on=onelist, how='left')
    # df = df.fillna(0)
    return df,test


def center(df):
    df['center manhattan distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 1 center x', 'pickup 1 center y']].reshape(1,-1),df.loc[x,['dropoff 1 center x', 'dropoff 1 center y']].reshape(1,-1),metric='manhattan')[0,0])
    df['center euclidean distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 1 center x', 'pickup 1 center y']].reshape(1,-1),df.loc[x,['dropoff 1 center x', 'dropoff 1 center y']].reshape(1,-1),metric='euclidean')[0,0])
    df['center cosine distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 1 center x', 'pickup 1 center y']].reshape(1,-1),df.loc[x,['dropoff 1 center x', 'dropoff 1 center y']].reshape(1,-1),metric='cosine')[0,0])
    # pick2 = pd.DataFrame(df.groupby(['pickup 1','pickup 2'])['pickup_x','pickup_y','pickup 1', 'pickup 2'].mean()).rename(columns={'pickup_x':'pickup 2 center x','pickup_y':'pickup 2 center y'})
    # df = df.merge(pick2, on=['pickup 1','pickup 2'], how='left')
    # dropoff2 = pd.DataFrame(df.groupby(['dropoff 1', 'dropoff 2'])['dropoff_x', 'dropoff_y', 'dropoff 1', 'dropoff 2'].mean()).rename(columns={'dropoff_x': 'dropoff 2 center x', 'dropoff_y': 'dropoff 2 center y'})
    # df = df.merge(dropoff2, on=['dropoff 1', 'dropoff 2'], how='left')
    # df['center 2 manhattan distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 2 center x', 'pickup 2 center y']].reshape(1,-1),df.loc[x,['dropoff 2 center x', 'dropoff 2 center y']].reshape(1,-1),metric='manhattan')[0,0])
    # df['center 2 euclidean distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 2 center x', 'pickup 2 center y']].reshape(1,-1),df.loc[x,['dropoff 2 center x', 'dropoff 2 center y']].reshape(1,-1),metric='euclidean')[0,0])
    # df['center 2 cosine distance'] = df.index.map(lambda x: metrics.pairwise.pairwise_distances(df.loc[x,['pickup 2 center x', 'pickup 2 center y']].reshape(1,-1),df.loc[x,['dropoff 2 center x', 'dropoff 2 center y']].reshape(1,-1),metric='cosine')[0,0])
    return df


def center2(df):
    df['manhattan mean speed 1'] =  df['clustering mean duration 1']/df['center manhattan distance']
    # df['manhattan mean speed 2'] = df['clustering mean duration 2']/df['center 2 manhattan distance']
    df['euclidean mean speed 1'] = df['clustering mean duration 1']/df['center euclidean distance']
    # df['euclidean mean speed 2'] = df['clustering mean duration 2']/df['center 2 euclidean distance']
    df['cosine mean speed 1'] = df['clustering mean duration 1']/df['center cosine distance']
    # df['cosine mean speed 2'] = df['clustering mean duration 2']/df['center 2 cosine distance']
    pca = PCA(n_components=1)
    df['pickup_cemter1_pca'] = pca.fit_transform(df[[  'pickup 1 center x', 'pickup 1 center y',]])
    df['dropoff_cemter1_pca'] = pca.fit_transform(df[[ 'dropoff 1 center x', 'dropoff 1 center y']])
    # df['pickup_cemter2_pca'] = pca.fit_transform(df[[  'pickup 2 center x', 'pickup 2 center y',]])
    # df['dropoff_cemter2_pca'] = pca.fit_transform(df[[ 'dropoff 2 center x', 'dropoff 2 center y']])
    return df
    




'##correlation'
def heat(df):
    corr = df.corr()
    # plt.figure(figsize=(18,18))
    # sns.heatmap(corr)
    # plt.show()


def pca(df):
    x = df[['pickup_x','pickup_y','dropoff_x','dropoff_y']].as_matrix()
    n_com = list(range(len(df.columns)-1))
    model = PCA()
    model.fit(x)
    # plt.plot(model.explained_variance_ratio_,linewidth=2)
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    # plt.show()


'##Feature Importance and Learning Curve'
def rf_importance(df):
    y = df[['duration']].values.reshape(-1,)
    x = df.drop(['duration'],axis=1).as_matrix()
    model = RandomForestRegressor(bootstrap=True,oob_score=True,n_estimators=500)
    model.fit(x, y)
    importance = model.feature_importances_
    importance = pd.DataFrame(importance, index= df.drop(['duration'],axis=1).columns, columns=["Importance"])
    importance["Std"] = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    xi = range(importance.shape[0])
    yi = importance.ix[:, 0]
    yerr = importance.ix[:, 1]
    # plt.bar(xi, yi, yerr=yerr, align="center")
    # plt.title('Feature Importance')
    # plt.show()
    return importance


def plot_learning_curve(df,model):
    y = df[['duration']].values.reshape(-1,)
    x = df.drop(['duration'],axis=1).as_matrix()
    model = model()
    train_sizes, train_scores, test_scores = learning_curve(model,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # plt.figure()
    # plt.title('Learning Curve')
    # plt.grid()
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    # plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    # plt.legend(loc="best")
    # plt.show()


def complete(df,test):
    df = df[df['duration'] >= df['duration'].quantile(0.01)]
    df = df[df['duration'] <= df['duration'].quantile(0.99)]
    df = clean(df)
    print('complete train')
    test = clean(test)
    df.to_csv('final1_train_step1.csv')
    test.to_csv('final1_test_step1.csv')
    print('complete clean')
    df,test= mini_finished(df,test,8)
    df.to_csv('final1_train_step2.csv')
    test.to_csv('final1_test_step2.csv')
    print('complete 1 cluster')
    df,test = second(df,test,8)
    print('complete 2 cluster')
    df,test=third(df,test,8)
    print('complete 3 cluster')
    df.to_csv('final1_train_step3.csv')
    print('train saved')
    test.to_csv('final1_test_step3.csv')
    print('test saved')
    df,test=group(df,test)
    print('complete group')
    df,test = group_duration(df,test)
    df.to_csv('final1_train_step4.csv')
    test.to_csv('final1_test_step4.csv')
    print('Complete center')
    df = center(df)
    test = center(test)
    df.to_csv('final1_train_step5.csv')
    test.to_csv('final1_test_step5.csv')
    print('Complete Group duation')
    df = center2(df)
    test = center2(test)
    print ('Complete center 2')
    df.to_csv('final1_train.csv')
    test.to_csv('final1_test.csv')
    return df,test

