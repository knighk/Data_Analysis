import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,make_scorer, accuracy_score,confusion_matrix,r2_score
from sklearn.model_selection import GridSearchCV,ShuffleSplit,learning_curve,cross_val_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures,RobustScaler
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.feature_selection import RFECV, SelectKBest,f_regression,chi2,mutual_info_regression
from sklearn.externals import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")


'##Set the WD path'
# datapath = '/Users/ace/Desktop/OneDrive - Georgia State University/ML Project/Gotham Cabs/Data'
datapath = 'E:\\OneDrive - Georgia State University\\ML Project\\Gotham Cabs\\Data\\gt'
# datapath = '/home/ubuntu/gt'

os.chdir(datapath)

# "##Raad Data set"

train = pd.read_csv('final1_train.csv',index_col=0)
train = train.sort_index()
test = pd.read_csv('final1_test.csv',index_col=0)
test = test.sort_index()





def plot_learning_curve(model,x,y,title):
    train_sizes, train_scores, test_scores = learning_curve(model,x,y,cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0),n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title(title)
    plt.ylim((0,1.2))
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best")


def select(df):
    y = df['duration'].values.ravel()
    x = df.drop(['duration'], axis=1).as_matrix()
    model = SelectFromModel(chi2,20)
    model.fit(x,y)
    cd = pd.DataFrame(df.drop('duration',axis=1).columns)
    cd['scores'] = model.scores_
    cd['pvalues'] = model.pvalues_
    return cd


# '##rf selected'
# ['Hour', 'braycurtis distance', 'pickup 1 Hour', 'pickup 3 Hour',
#        'dropoff 3 Hour', 'center manhattan distance',
#        'clustering mean duration 1', 'clustering mean duration 3',
#        'Day clustering mean duration 1', 'pickup 2 Hour',
#        'Year clustering mean duration 1', 'Minutes','pickup_x'
#        'cosine Min distance', 'chebyshev Min distance', 'dropoff 2 Hour',
#        'clustering mean duration 2', 'Month clustering mean duration 1',
#        'pickup_y', 'dropoff_y', 'canberra distance', 'chebyshev distance',
#        'pickup_pca', 'dropoff_pca', 'canberra mean speed 3', 'rides',
#        'Distance', 'dropoff 1 Day', 'center 3 manhattan distance',
#        'canberra mean speed 2', 'dropoff_x', 'Day', 'l2 Min distance',
#        'cosine distance', 'canberra Min distance', 'pickup 1 Day',
#        'pickup 2 Day', 'dropoff 2 Day', 'pickup 3 Day', 'dropoff 3 Day',
#        'center 2 canberra distance']


def rf(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    rfe = RFECV(RandomForestRegressor(bootstrap=True,n_jobs=n_jobs), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rfe.fit(x,y)
    x = rfe.transform(x)
    test = rfe.transform(test.as_matrix())
    x= RobustScaler().fit_transform(x,y)
    test = RobustScaler().fit_transform(test)
    print('Complete StandardScaler')
    param_grid = {'max_features':['auto'], 'max_depth': list(np.linspace(41,43,3,dtype=int)),'n_estimators':[30],'bootstrap':[True]}
    model = GridSearchCV(RandomForestRegressor(),cv= ShuffleSplit(n_splits=2, test_size=0.2, random_state=0) ,param_grid=param_grid,scoring=make_scorer(r2_score,greater_is_better=True),verbose=1,return_train_score=True,n_jobs=n_jobs)
    # model = RandomForestRegressor(bootstrap=True)
    model.fit(x,y)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('rf_pre.csv')
    plot_learning_curve(best,x,y,'Random Forest Learning Curve')
    joblib.dump(model,'rf_train_cv.m')
    joblib.dump(best,'rf_best.m')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    plt.savefig('Random Forest Learning Curve.png')
    plt.show()
    print('MSE for Random Forest Regression is', mse)
    print('Score for Random Forest Regression  is', msc)
    return mse,msc,bs



def ada(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    rfe = RFECV(RandomForestRegressor(bootstrap=True,n_jobs=n_jobs), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rfe.fit(x,y)
    x = rfe.transform(x)
    test = rfe.transform(test.as_matrix())
    x= RobustScaler().fit_transform(x,y)
    test = RobustScaler().fit_transform(test)
    print('Complete StandardScaler')
    base = AdaBoostRegressor
    param_grid = {'base_estimator':[RandomForestRegressor()],'learning_rate': np.linspace(0.1, 2, 10),'loss': ['linear', 'square', 'exponential']}
    model = GridSearchCV(base(),cv= ShuffleSplit(n_splits=2, test_size=0.2, random_state=0) ,param_grid=param_grid,scoring=make_scorer(r2_score,greater_is_better=True),verbose=1,return_train_score=True,n_jobs=n_jobs)
    # model = RandomForestRegressor(bootstrap=True)
    model.fit(x,y)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('ada_pre.csv')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    plot_learning_curve(best,x,y,'Adaboost Learning Curve')
    joblib.dump(model,'ada_train_cv.m')
    joblib.dump(best,'ada_train.m')
    plt.savefig('ADA Learning Curve.png')
    plt.show()
    print('MSE for ADA Regression is', mse)
    print('Score for ADA Regression  is', msc)
    return mse,msc,bs


def br(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    x= RobustScaler().fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    base = BaggingRegressor
    param_grid = {'base_estimator':[RandomForestRegressor()],'n_estimators': [100],'bootstrap':[True],'bootstrap_features':[True],'max_samples':np.linspace(0.1,1,5)}
    model = GridSearchCV(base(),cv= ShuffleSplit(n_splits=2, test_size=0.2, random_state=0) ,param_grid=param_grid,scoring=make_scorer(r2_score,greater_is_better=True),verbose=1,return_train_score=True,n_jobs=n_jobs)
    # model = RandomForestRegressor(bootstrap=True)
    model.fit(x_train,y_train)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('br_pre.csv')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    joblib.dump(model,'BAG_train_cv.m')
    joblib.dump(best,'BAG_train.m')
    plot_learning_curve(base(**bs),x,y,'Bagging Learning Curve')
    plt.savefig('Bagging Learning Curve.png')
    plt.show()
    print('MSE for Bagging Regression is', mse)
    print('Score for Bagging Regression  is', msc)
    return mse,msc,bs


# '##gbdt selected'
# ['Hour', 'braycurtis distance', 'pickup 1 Hour', 'pickup 3 Hour',
#        'dropoff 3 Hour', 'center manhattan distance',
#        'clustering mean duration 1', 'clustering mean duration 3',
#        'Day clustering mean duration 1',
#        'Year clustering mean duration 1', 'pickup_y', 'dropoff_y','pickup_x','dropoff_x'
#        'canberra distance', 'chebyshev distance', 'pickup_pca',
#        'dropoff_pca', 'canberra mean speed 3', 'Weekday(num)',
#        'Manhatan Distance', 'dropoff 1 Hour', 'Log Haversine Distance',
#        'manhattan distance', 'Month', 'Haversine', 'dropoff 1 Year',
#        'center canberra distance', 'canberra mean speed 1',
#        'L1 distance']

def gbdt(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    rfe = RFECV(GradientBoostingRegressor(), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rfe.fit(x,y)
    x = rfe.transform(x)
    test = rfe.transform(test.as_matrix())
    x= RobustScaler().fit_transform(x,y)
    test = RobustScaler().fit_transform(test)
    param_grid = {'max_features':['auto'], 'max_depth':list(np.linspace(11,13,3,dtype=int)),'min_samples_split':list(np.linspace(45,55,3,dtype=int)), 'min_samples_leaf':list(np.linspace(19,21,3,dtype=int)),'learning_rate': [0.01,0.05,0.1],'warm_start':[True]}
    model = GridSearchCV(GradientBoostingRegressor(),cv= ShuffleSplit(n_splits=2, test_size=0.2, random_state=0) ,param_grid=param_grid,scoring=make_scorer(r2_score,greater_is_better=True),verbose=1,return_train_score=True,n_jobs=n_jobs)
    # model = RandomForestRegressor(bootstrap=True)
    model.fit(x,y)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('gbdt_pre.csv')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    plot_learning_curve(best,x,y,'GBDT Learning Curve')
    joblib.dump(model,'GBDT_train_cv.m')
    joblib.dump(best,'GBDT_train.m')
    plt.savefig('GBDT Learning Curve.png')
    plt.show()
    print('MSE for GBDT is', mse)
    print('Score for GBDT  is', msc)
    return mse,msc,bs


# '##edt selected'
# ['Hour', 'braycurtis distance', 'pickup 1 Hour', 'pickup 3 Hour',
#        'dropoff 3 Hour', 'center manhattan distance',
#        'clustering mean duration 1', 'clustering mean duration 3',
#        'Day clustering mean duration 1', 'pickup 2 Hour',
#        'Year clustering mean duration 1', 'Minutes',
#        'cosine Min distance', 'chebyshev Min distance', 'dropoff 2 Hour',
#        'clustering mean duration 2', 'Month clustering mean duration 1',
#        'Weekday(num)', 'Manhatan Distance', 'dropoff 1 Hour',
#        'braycurtis mean speed 3', 'Log Haversine Distance',
#         'Pick Up Minute of the Day','pickup_x','pickup_y','dropoff_x','dropoff_y','pickup_pca','dropoff_pca',
#        'L1 distance', 'pickup 2 Minutes_bins',
#        'dropoff 2 Minutes_bins']

def edt(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    rfe = RFECV(ExtraTreesRegressor(n_jobs=n_jobs), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rfe.fit(x,y)
    x = rfe.transform(x)
    test = rfe.transform(test.as_matrix())
    x= RobustScaler().fit_transform(x,y)
    test = RobustScaler().fit_transform(test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    param_grid = {'max_features':['auto'], 'max_depth': list(np.linspace(29,32,3,dtype=int)),'min_samples_split':list(np.linspace(53,55,3,dtype=int)),'min_impurity_decrease': list(np.linspace(0.01,0.1,3)),'warm_start':[True]}
    model = GridSearchCV(ExtraTreesRegressor(),cv= ShuffleSplit(n_splits=2, test_size=0.2, random_state=0) ,param_grid=param_grid,scoring=make_scorer(r2_score,greater_is_better=True),verbose=1,return_train_score=True,n_jobs=n_jobs)
    # model = RandomForestRegressor(bootstrap=True)
    model.fit(x_train,y_train)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('edt_pre.csv')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    plot_learning_curve(best,x,y,'Extra Decisition Tree Learning Curve')
    joblib.dump(model,'edt_train_cv.m')
    joblib.dump(best,'edt_train.m')
    plt.savefig('Extra Tree Learning Curve.png')
    plt.show()
    print('MSE for Extra Tree is', mse)
    print('Score for Extra Tree  is', msc)
    return mse,msc,bs

'##xgb selected'
# ['Hour', 'braycurtis distance', 'pickup 1 Hour', 'pickup 3 Hour',
#        'dropoff 3 Hour', 'center manhattan distance',
#        'clustering mean duration 1', 'clustering mean duration 3',
#        'Day clustering mean duration 1', 'pickup 2 Hour', 'pickup_y',
#        'dropoff_y', 'canberra distance', 'chebyshev distance',
#        'pickup_pca', 'dropoff_pca', 'canberra mean speed 3', 'rides',
#        'Distance', 'dropoff 1 Day', 'center 3 manhattan distance',
#        'canberra mean speed 2', 'Weekday(num)', 'Manhatan Distance',
#        'dropoff 1 Hour', 'braycurtis mean speed 3', 'Month', 'Haversine',
#        'dropoff 1 Year', 'center canberra distance',
#        'canberra mean speed 1', 'Day of Year', 'pickup 1', 'dropoff 3',
#        'dropoff 1 Month', 'dropoff 2 center x', 'pickup 3 center y',
#        'dropoff 3 center x', 'center braycurtis distance',
#        'center 3 braycurtis distance']
def xgboost(df,test):
    y = df['duration'].values.ravel()
    x = df.drop('duration',axis=1).as_matrix()
    base = XGBRegressor
    rfe = RFECV(base(n_jobs=n_jobs), cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rfe.fit(x,y)
    x = rfe.transform(x)
    test = rfe.transform(test.as_matrix())
    x= RobustScaler().fit_transform(x,y)
    test = RobustScaler().fit_transform(test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    param_grid = {'learning_rate': list(np.linspace(0.2,0.2,1)), 'min_child_weight':list(np.linspace(20,30,2,dtype=int)), 'max_depth': list(np.linspace(15,15,1,dtype=int)),'reg_lambda':list(np.linspace(5,5,1,dtype=int)),'n_estimators':[1000,500]}
    model = GridSearchCV(base(),cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0), param_grid=param_grid,scoring=make_scorer(r2_score, greater_is_better=True), verbose=1,return_train_score=True,n_jobs=n_jobs)
    model.fit(x_train,y_train)
    bs = model.best_params_
    best = model.best_estimator_
    y_pre = best.predict(test)
    y_pre = pd.DataFrame(y_pre,columns=['duration'])
    y_pre.to_csv('xgb1_pre.csv')
    mse = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=n_jobs)
    msc = cross_val_score(best,x,y,cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),scoring=make_scorer(r2_score, greater_is_better=True),n_jobs=n_jobs)
    # plot_learning_curve(best,x,y,'XGB Learning Curve')
    joblib.dump(model,'XGB_cv.m')
    joblib.dump(best,'XGB_train.m')
    # plt.savefig('XGBoost Learning Curve.png')
    # plt.show()
    print('Score for xgboost is', msc)
    print('MSE for xgboost is', mse)
    return mse,msc,bs






def test_all(df,test):
    df = df.sort_index()
    test = test.sort_index()
    tem = pd.DataFrame()
    # t1, t2,t3= xgboost(df,test)
    # tem.loc[1,'Model'] = 'Xgboost'
    # tem.loc[1,'mse'] = str(t1)
    # tem.loc[1,'msc'] = str(t2)
    # tem.loc[1,'Best Param'] = str(t3)
    # tem.to_csv('tem test.csv')
    # t1, t2,t3= gbdt(df,test)
    # tem.loc[2,'Model'] = 'GBDT'
    # tem.loc[2,'mse'] = str(t1)
    # tem.loc[2,'msc'] = str(t2)
    # tem.loc[2,'Best Param'] = str(t3)
    # tem.to_csv('tem test.csv')
    t1, t2,t3 = edt(df,test)
    tem.loc[3,'Model'] = 'EDT'
    tem.loc[3,'mse'] = str(t1)
    tem.loc[3,'msc'] = str(t2)
    tem.loc[3,'Best Param'] = str(t3)
    tem.to_csv('tem test.csv')
    t1, t2,t3 = rf(df,test)
    tem.loc[4,'Model'] = 'Random Forest'
    tem.loc[4,'mse'] = str(t1)
    tem.loc[4,'msc'] = str(t2)
    tem.loc[4,'Best Param'] = str(t3)
    tem.to_csv('tem test.csv')
    t1, t2,t3= ada(df,test)
    tem.loc[5,'Model'] = 'Adaboost'
    tem.loc[5,'mse'] = str(t1)
    tem.loc[5,'msc'] = str(t2)
    tem.loc[5,'Best Param'] = str(t3)
    tem.to_csv('tem test.csv')



def rfe(df):
    y = df['duration'].values.ravel()
    x = df.drop(['duration'], axis=1).as_matrix()
    rf = RFECV(RandomForestRegressor(bootstrap=True,n_jobs=n_jobs), cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    rf.fit(x,y)
    cd = pd.DataFrame(df.drop('duration',axis=1).columns)
    cd['RF Selected'] = rf.support_
    cd['RF ranking'] = rf.ranking_
    ed = RFECV(ExtraTreesRegressor(bootstrap=True,n_jobs=n_jobs), cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
               scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, step=int(0.1 * x.shape[1]),
               n_jobs=n_jobs)
    ed.fit(x,y)
    cd['EDT Selected'] = ed.support_
    cd['EDT ranking'] = ed.ranking_
    base = XGBRegressor
    xg = RFECV(base(),cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),verbose=2,step=int(0.1*x.shape[1]),n_jobs=n_jobs)
    xg.fit(x,y)
    cd['XGboost Selected'] = xg.support_
    cd['XGboost ranking'] = xg.ranking_
    gdb = RFECV(GradientBoostingRegressor(),cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),scoring=make_scorer(mean_squared_error,greater_is_better=False),verbose=2,step=int(0.1*x.shape[1]),n_jobs=n_jobs)
    gdb.fit(x,y)
    cd['GBDT Selected'] = gdb.support_
    cd['GBDT ranking'] = gdb.ranking_
    cd.to_csv('cd.csv')
    return cd