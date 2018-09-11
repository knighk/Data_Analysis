"""## Import """
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Convolution1D,MaxPooling1D,AveragePooling1D,SeparableConv1D
from keras.layers.normalization import BatchNormalization
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from keras.layers.noise import GaussianNoise
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.metrics import roc_auc_score,precision_score,classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from keras.layers import Input,Dense
from scipy import interp
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
import gc
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from keras import losses
from keras import metrics
from keras.callbacks import Callback, EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import ParameterGrid
import os
import keras
import sklearn
from sklearn.utils import resample


"""#change the system path.Change to your system path if you need to run the script"""
os.getcwd()
os.chdir('E:\\OneDrive - Georgia State University\\Kaggle Home Equity\\Ping\\Data')

"""#read the data set"""
final_train = pd.read_csv('E:\\OneDrive - Georgia State University\\Kaggle Home Equity\\Final Train Test\\Final Train.csv')
final_train = final_train.drop(['NAME_FAMILY_STATUS_Unknown','NAME_INCOME_TYPE_Maternity leave'],axis=1)
"""##Up Sample data set"""
n = len(final_train.columns)-2

def balance(final_train):
    l = final_train['TARGET'].value_counts()
    n = l[0]
    data_majority = final_train[final_train['TARGET']==0]
    data_minority = final_train[final_train['TARGET']==1]
    df_minority_upsampled = resample(data_minority,n_samples=n,replace=True,random_state=123)
    df_upsampled = pd.concat([data_majority,df_minority_upsampled])
    return df_upsampled

class Bp:
    def __init__(self, df):
        self.df = df.drop('SK_ID_CURR',axis=1).reset_index(drop=True)
        self.shape = self.df.drop('TARGET', axis=1).shape
        self.dim = len(self.df.columns)-1
        self.model_x = StandardScaler()
        self.df.loc[:,self.df.columns !='TARGET']= self.model_x.fit_transform(self.df.loc[:,self.df.columns !='TARGET'])
        self.train, self.test = train_test_split(self.df, test_size=0.01)
        self.n0 = self.train['TARGET'].value_counts()[0]
        self.n1 = self.train['TARGET'].value_counts()[1]
        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)
        self.x_test = self.test.drop('TARGET', axis=1)
        self.y_test = self.test['TARGET']
        self.x_train = self.train.drop('TARGET',axis = 1)
        self.y_train = self.train['TARGET']
        self.x_train,self.y_train = SMOTE(n_jobs=7,ratio={0:self.n0,1:self.n1*7}).fit_sample(self.x_train,self.y_train)

    def as_keras_metric(self,method):
        import functools
        from keras import backend as K
        import tensorflow as tf
        @functools.wraps(method)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = method(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value
        return wrapper

    def bpmodel(self,dic= {'act':'sigmoid','init':'lecun_normal'},nl={'node':360},learate=0.1,name='Default'):
        gc.collect()
        act = dic['act']
        init = dic['init']
        number = nl['node']
        model = Sequential()
        model.add(Dense(number,input_dim=self.dim,kernel_initializer=init,kernel_regularizer=regularizers.l1_l2(l1=0.005,l2=0.005)))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(number*10,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(number*8,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dropout(0.2))
        model.add(Dense(number*6,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(number*4,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dropout(0.2))
        model.add(Dense(number*2,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(number,kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(int(number/2),kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dropout(0.2))
        model.add(Dense(int(number/4),kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(int(number/6),kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(int(number/12),kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(Dense(1,activation='sigmoid',kernel_initializer=init))
        auc_roc = self.as_keras_metric(tf.metrics.auc)
        op = optimizers.adamax(lr=learate)
        pr = self.as_keras_metric(tf.metrics.precision)
        rec = self.as_keras_metric(tf.metrics.recall)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=[auc_roc,rec,pr])
        model.save('E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Code\\bpmodel.h5')
        # es = EarlyStopping(monitor='auc',patience=9, verbose=1,mode='max')
        # lr = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=3, verbose=1, mode='max', min_delta=0.0001,cooldown=0, min_lr=0)
        # history = model.fit(self.x_train, self.y_train,epochs=50,batch_size=1000,callbacks=[es,lr],validation_split=0.1)
        # y_pre = model.predict(self.x_test,batch_size=1000)
        # y_pre1 = model.predict_classes(self.x_test,batch_size=1000)
        # score = model.evaluate(self.x_test,self.y_test,batch_size=1000)
        # f1 = (2*score[3]*score[2])/(score[3]+score[2])
        # plt.figure(figsize=(12,6))
        # plt.subplot(121)
        # plt.plot(history.history['auc'])
        # plt.plot(history.history['val_auc'])
        # plt.title('model auc')
        # plt.ylabel('auc')
        # plt.xlabel('epoch')
        # plt.suptitle('BP: The Over Sampling is %s'%name)
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.subplot(122)
        # plt.plot(history.history['precision'])
        # plt.plot(history.history['val_precision'])
        # plt.title('model precision')
        # plt.ylabel('precision')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig('E:\\OneDrive - Georgia State University\\Kaggle Home Equity\\Ping\\Reports\\DeepLearning\\BP\\%s.png'%name)
        # plt.show()
        # print('The test AUC is ',score[1],'The recall is',score[2],'The precision is',score[3])
        # print('F1 Score is',f1)
        return model

    def cnn(self,dic= {'act':'tanh','init':'lecun_normal'},number={'kernel':n,'pool':100},learate=0.01,name='Default',ndim = 256):
        gc.collect()
        act = dic['act']
        init = dic['init']
        kernel = number['kernel']
        pool = number['pool']
        model = Sequential()
        model.add(Convolution1D(ndim,kernel_size=(kernel),input_shape=(self.x_train.shape[1],1),padding='same',kernel_initializer=init,kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01)))
        model.add(BatchNormalization())
        model.add(Activation(act))
        model.add(MaxPooling1D(pool_size=(pool),padding='same'))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(GaussianNoise(0.1))
        model.add(Dense(int(ndim/2)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(GaussianNoise(0.1))
        model.add(Dense(int(ndim/2)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(int(ndim/4)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(int(ndim/4)))
        model.add(Activation(act))
        model.add(BatchNormalization())
        model.add(Dense(1,activation='sigmoid'))
        auc_roc = self.as_keras_metric(tf.metrics.auc)
        op = optimizers.adamax(lr=learate)
        pr = self.as_keras_metric(tf.metrics.precision)
        rec = self.as_keras_metric(tf.metrics.recall)
        model.compile(loss='binary_crossentropy', optimizer=op, metrics=[auc_roc,rec,pr])
        # es = EarlyStopping(monitor='auc',patience=9, verbose=1,mode='max')
        # lr = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=3, verbose=1, mode='max', min_delta=0.0001,cooldown=0, min_lr=0)
        # history = model.fit(self.x_train, self.y_train, epochs=30, batch_size=250, callbacks=[es, lr], validation_split=0.1,shuffle=True)
        # y_pre = model.predict_classes(self.x_test,batch_size=250)
        # score = model.evaluate(self.x_test,self.y_test,batch_size=250)
        # f1 = (2*score[3]*score[2])/(score[3]+score[2])
        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        # plt.plot(history.history['auc'])
        # plt.plot(history.history['val_auc'])
        # plt.title('model auc')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.suptitle('CNN:The Over Sampling is %s'%name)
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.subplot(122)
        # plt.plot(history.history['precision'])
        # plt.plot(history.history['val_precision'])
        # plt.title('model precision')
        # plt.ylabel('precision')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig('E:\\OneDrive - Georgia State University\\Kaggle Home Equity\\Ping\\Reports\\DeepLearning\\CNN\\%s.png'%name)
        # plt.show()
        # print('The test AUC is ',score[1],'The recall is',score[2],'The precision is',score[3])
        # print('F1 Score is',f1)
        return model

def oversample():
    # kind = ['regular','borderline1','borderline2','svm']
    # for x in kind:
    #     sm = SMOTE(kind=x)
    #     bp = Bp(final_train,sm)
    #     bp.bpmodel(name=x)
    #     bp.cnn(name=x)
    ada = ADASYN()
    bp = Bp(final_train,ada)
    bp.bpmodel(name='ADASYN')
    bp.cnn(name='ADASYN')

def test():
    bp = Bp(final_train)
    param_grid ={'act':['elu','selu','softplus','relu','tanh','sigmoid'], 'init':['orthogonal','glorot_normal','he_normal','lecun_normal',]}
    param = ParameterGrid(param_grid)
    number_grid = {'kernel':[n,int(n/2),int(n/4),100],'pool':[int(n/2),int(n/4),100,50,20]}
    number = ParameterGrid(number_grid)
    global l1
    l1 = pd.DataFrame(columns=('init','act'))
    l2 = pd.DataFrame(columns=('kernel','pool'))
    l3 = pd.DataFrame(columns=('Node#','model'))
    i = 0
    # for x in param:
    #     score,f1 = bp.bpmodel(x)
    #     x['auc'] = score[1]
    #     x['recall'] = score[2]
    #     x['precision'] = score[3]
    #     x['F1 Score'] = f1
    #     x['model'] = 'BP'
    #     df = pd.DataFrame(x,index=[i])
    #     l1 = l1.append(df,sort=False,ignore_index=True)
    #     i+=1
    # i = 0
    # for x in param:
    #     score,f1 = bp.cnn(dic=x)
    #     x['auc'] = score[1]
    #     x['recall'] = score[2]
    #     x['precision'] = score[3]
    #     x['F1 Score'] = f1
    #     x['model'] = 'CNN'
    #     df = pd.DataFrame(x,index=[i])
    #     l1 = l1.append(df,sort=False,ignore_index=True)
    #     i+=1
    # l1.to_csv('Param Choose.csv')
    # i = 0
    # for x in number:
    #     score,f1 = bp.cnn(number=x)
    #     x['auc'] = score[1]
    #     x['recall'] = score[2]
    #     x['precision'] = score[3]
    #     x['F1 Score'] = f1
    #     x['model'] = 'CNN'
    #     df = pd.DataFrame(x,index=[i])
    #     l2 = l2.append(df,sort=False,ignore_index=True)
    #     i+=1
    # l2.to_csv('Number Choose.csv')
    # bp_gird = {'node':[n*4,n*2,n,int(n/2),int(n/4),50]}
    # bp_num = ParameterGrid(bp_gird)
    # for x in bp_num:
    #     score,f1 = bp.bpmodel(nl=x)
    #     x['auc'] = score[1]
    #     x['recall'] = score[2]
    #     x['precision'] = score[3]
    #     x['F1 Score'] = f1
    #     x['model'] = 'bp'
    #     df = pd.DataFrame(x,index=[i])
    #     l3 = l3.append(df,sort=False,ignore_index=True)
    #     i+=1
    # for x in opti:
    #     bp.bpmodel(op=x)
    #     bp.cnn(op=x)
    lr = [1,0.5,0.1,0.05,0.01]
    for x in lr:
        bp.bpmodel(learate=x)
        bp.cnn(learate=x)
    return l1

def cv():
    bp = Bp(final_train)
    es = EarlyStopping(monitor='auc',patience=9, verbose=1,mode='max')
    lr = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=3, verbose=1, mode='max', min_delta=0.0001,cooldown=0, min_lr=0)
    cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    bp.x_train = bp.x_train.reshape(bp.x_train.shape[0], bp.x_train.shape[1], 1)
    for train, test in cv.split(bp.x_train,bp.y_train):
        model = KerasClassifier(build_fn=bp.cnn, batch_size=250, epochs=30, callbacks=[es, lr],validation_split=0.1)
        model.fit(bp.x_train[train], bp.y_train[train])
        probas_ = model.predict_proba(bp.x_train[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(bp.y_train[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix,')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion():
    name = 'BP'
    bp = Bp(final_train)
    unique, counts = np.unique(bp.y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    # bp.x_train = bp.x_train.reshape(bp.x_train.shape[0], bp.x_train.shape[1], 1)
    # bp.x_test = bp.x_test.values.reshape(bp.x_test.shape[0], bp.x_test.shape[1], 1)
    es = EarlyStopping(monitor='auc',patience=10, verbose=1,mode='max')
    lr = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=3, verbose=1, mode='max', min_delta=0.0001,cooldown=0, min_lr=0)
    model = KerasClassifier(build_fn=bp.bpmodel,batch_size=5000, epochs=50, callbacks=[es, lr],validation_split=0.1,shuffle=True)
    model.fit(bp.x_train, bp.y_train,batch_size = 5000)
    y_pre = model.predict(bp.x_test,batch_size=5000)
    cm = confusion_matrix(bp.y_test,y_pre)
    class_names = ['0','1']
    plot_confusion_matrix(cm, classes=class_names,title='Outsample Test %s ,'%(name))
    plt.savefig('E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Reports\DeepLearning\Sampling\cm%s.png'%(name))
    plt.show()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    probas_ = model.predict_proba(bp.x_test)
    fpr, tpr, thresholds = roc_curve(bp.y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='(AUC = %0.2f)' % (roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('OutSample Test %s'%(name))
    plt.legend(loc="lower right")
    plt.savefig('E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Reports\DeepLearning\Sampling\\roc%s.png'%(name))
    plt.show()
    plt.figure()
    average_precision = average_precision_score(bp.y_test, probas_[:, 1])
    precision, recall, _ = precision_recall_curve(bp.y_test, probas_[:, 1])
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OutSample Test %s,AP={%0.2f}'%(name,average_precision))
    plt.savefig('E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Reports\DeepLearning\Sampling\\prc%s.png'%(name))
    plt.show()
    nlist = final_train.drop(['TARGET','SK_ID_CURR'],axis=1).columns
    df_test = pd.read_csv('E:\OneDrive - Georgia State University\Kaggle Home Equity\\Nan\Data\\test_k_raw (1).csv')
    df_result = pd.DataFrame()
    df_result['SK_ID_CURR'] = df_test['SK_ID_CURR']
    df_test = df_test.fillna(-1)
    df_test = df_test.set_index('SK_ID_CURR',drop=True)
    df_test = df_test[nlist]
    df_test = bp.model_x.transform(df_test)
    # df_test = df_test.reshape(df_test.shape[0], df_test.shape[1], 1)
    y_true = model.predict(df_test,batch_size=5000)
    y_pro = model.predict_proba(df_test,batch_size=5000)
    if len(y_true) ==len(df_test):
        df_result['TARGET'] = y_pro[:,1]
        df_result.to_csv('E:\OneDrive - Georgia State University\Kaggle Home Equity\Ping\Result\\bpprob%0.2f.csv'%roc_auc,index=False)
        print('The Test Rows are correct')
    else:
        print('There are missing on test result')
    return y_true,y_pro


def grid():
    bp = Bp(final_train)
    bp.x_train = bp.x_train.reshape(bp.x_train.shape[0], bp.x_train.shape[1], 1)
    bp.x_test = bp.x_test.values.reshape(bp.x_test.shape[0], bp.x_test.shape[1], 1)
    es = EarlyStopping(monitor='auc',patience=9, verbose=1,mode='max')
    lr = ReduceLROnPlateau(monitor='auc', factor=0.1, patience=3, verbose=1, mode='max', min_delta=0.0001,cooldown=0, min_lr=0)
    model = KerasClassifier(build_fn=bp.cnn)
    param_grid = {'batch_size':[250,150,100,50,30,10],'callbacks':[[es,lr]],'epochs':[30,50,100]}
    grid = GridSearchCV(model,param_grid=param_grid,cv=2,scoring='roc_auc',verbose=1)
    grid.fit(bp.x_train, bp.y_train)
    return grid.best_params_,grid.best_score_

def sampling():
    nlist = list(range(45,56))
    param_grid = {'knn': nlist}
    param = ParameterGrid(param_grid)
    for x in param:
        gc.collect()
        plot_confusion(dic=x)
    # nlist = [512,256,128]
    # for x in nlist:
    #     gc.collect()
    #     plot_confusion(knn=x)

y_tru,y_pro = plot_confusion()
