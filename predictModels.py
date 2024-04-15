import pandas as pd
import numpy as np
from os import walk
from itertools import chain
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
%matplotlib inline
from pandas.core.frame import DataFrame
#from sklearn.preprocessing import MinMaxScaler
#from keras.optimizers import SGD
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from keras.layers import GRU
from statsmodels.tsa.arima_model import ARIMA
import time
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF 
from statsmodels.stats.diagnostic import acorr_ljungbox
 

#抓地面檔資料
def Ground_Read(path):
    list1_Ground = []
    Ground = []
    
    #抓資料
    ascii_Ground = np.loadtxt(path, skiprows=6)
    #拆分
    ascii_Ground = np.array_split(ascii_Ground, 62, axis=0)
    for i in range(62):
        list1_Ground.append(np.array_split(ascii_Ground[i], 55, axis=1))
    #地面檔1:1資料
    sum_temp = []
    ground_mean = 0

    Ground = list(chain(*list1_Ground[34][43]))
    #計算網格平均值
    for i in range(len(Ground)):
        if Ground[i]>0:
            sum_temp.append(Ground[i])
    if sum(sum_temp) > 0:
        ground_mean = np.mean(sum_temp)                  
    #print(Ground[0][0])
    split_gruond.append(ground_mean)


#抓雨量檔資料
def Railfall_Read(path):
    #抓資料
    ascii_Rainfall = np.loadtxt(path, skiprows=6)
    split_rainfall.append(ascii_Rainfall[34][43])


#建Training Data
def buildTrain(split_rainfall, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(split_rainfall.shape[0]-futureDay-pastDay):
        X_train.append(np.array(split_rainfall.iloc[i:i+pastDay]))
        Y_train.append(np.array(split_rainfall.iloc[i+pastDay:i+pastDay+futureDay]["Ground"]))
    return np.array(X_train), np.array(Y_train)


#切Training data,Validation data,Test data
def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    X_test = np.array_split(X_val, 2, axis=0)[1]
    Y_test = np.array_split(Y_val, 2, axis=0)[1]
    X_val = np.array_split(X_val, 2, axis=0)[0]
    Y_val = np.array_split(Y_val, 2, axis=0)[0]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


#建立Model
#GBTR
def GradientBoostingRegressor_Model(X_train, Y_train, X_test, Y_test):
    #model = GradientBoostingRegressor()
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=10, random_state=7, loss='ls')
    model.fit(X_train,Y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    #評估模型 - R平方
    print("The R2 score on the Train set is:\\t{:0.3f}".format(r2_score(Y_train, y_pred_train)))
    print("The R2 score on the Test set is:\\t{:0.3f}".format(r2_score(Y_test, y_pred_test)))    
    #評估模型 - MAE
    print('MAE Train：\\t{:0.3f}'.format(mean_absolute_error(Y_train, y_pred_train)))
    print('MAE Test：\\t{:0.3f}'.format(mean_absolute_error(Y_test, y_pred_test)))    
    #評估模型 - RMSE
    print('RMSE Train：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_train, y_pred_train))))
    print('RMSE Test：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_test, y_pred_test))))
    OthersModel_Score(Y_test,y_pred_test)
        
def SVM_Model(X_train, Y_train, X_test, Y_test):
    # 線性核函數
    model = SVR(kernel="rbf")
    model.fit(X_train, Y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    #評估模型 - R平方
    print("The R2 score on the Train set is:\\t{:0.3f}".format(r2_score(Y_train, y_pred_train)))
    print("The R2 score on the Test set is:\\t{:0.3f}".format(r2_score(Y_test, y_pred_test)))    
    #評估模型 - MAE
    print('MAE Train：\\t{:0.3f}'.format(mean_absolute_error(Y_train, y_pred_train)))
    print('MAE Test：\\t{:0.3f}'.format(mean_absolute_error(Y_test, y_pred_test)))    
    #評估模型 - RMSE
    print('RMSE Train：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_train, y_pred_train))))
    print('RMSE Test：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_test, y_pred_test))))
    OthersModel_Score(Y_test,y_pred_test)


def ARIMA_Model(train_norm):    
    X = train_norm['Ground'].values
    size = int(len(X) * 0.7)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    
    model_train = ARIMA(history, order=(1,0,0))
    result1 = model_train.fit() 
    result1.summary()  
    train_predictions = result1.predict(1, len(train) , typ = 'levels')
    error = mean_squared_error(train, train_predictions) #第一個值NAN
    print('Test MSE: %.30f' % error)
    
    for t in range(len(test)):
        model = ARIMA(history, order=(1,0,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        predictions.append(output[0])
        history.append(test[t])
        if (t%5000)==0:
            print(t)
    error = mean_squared_error(test, predictions) #第一個值NAN
    print('Test MSE: %.30f' % error)
    data={"predictions" : predictions}#轉成字典
    data_df=DataFrame(data)#轉成DataFrame
    data_df.to_csv("C:/Flooding/data/ARIMA-predictions.csv",index=False,sep=',') #匯出CSV
    
    
    predictions = pd.read_csv("C:/Flooding/data/ARIMA-predictions.csv") #讀取CSV檔   

    #評估模型 - R平方
    print("The R2 score on the Train set is:\\t{:0.3f}".format(r2_score(train, train_predictions)))
    print("The R2 score on the Test set is:\\t{:0.3f}".format(r2_score(test, predictions)))
    #評估模型 - MAE
    print('MAE Train：\\t{:0.30f}'.format(mean_absolute_error(train, train_predictions)))
    print('MAE Test：\\t{:0.30f}'.format(mean_absolute_error(test, predictions)))    
    #評估模型 - RMSE
    print('RMSE Train：\\t{:0.30f}'.format(np.sqrt(mean_squared_error(train, train_predictions))))
    print('RMSE Test：\\t{:0.30f}'.format(np.sqrt(mean_squared_error(test, predictions))))    
    
    #淹水級距分類
    Y_test = test
    y_pred_test = predictions['predictions']
    lab_test = []
    lab_pred = []
    for i in range(len(Y_test)):
        if 0.1>=Y_test[i]>=0:lab_test.append('0')
        elif 0.3>=Y_test[i]>0.1:lab_test.append('1')
        elif 0.5>=Y_test[i]>0.3:lab_test.append('2')
        elif 1>=Y_test[i]>0.5:lab_test.append('3')
        elif 2>=Y_test[i]>1:lab_test.append('4')
        elif Y_test[i]>2:lab_test.append('5')
        else:lab_test.append('-1')
    for i in range(len(y_pred_test)):
        if 0.1>=y_pred_test[i]>=0:lab_pred.append('0')
        elif 0.3>=y_pred_test[i]>0.1:lab_pred.append('1')
        elif 0.5>=y_pred_test[i]>0.3:lab_pred.append('2')
        elif 1>=y_pred_test[i]>0.5:lab_pred.append('3')
        elif 2>=y_pred_test[i]>1:lab_pred.append('4')
        elif y_pred_test[i]>2:lab_pred.append('5')
        else:lab_pred.append('-1')        
    
    target_names = ['-1','0','1' ,'2','3','4','5']
    print('準確率：', accuracy_score(lab_test, lab_pred))
    print("report:\n",classification_report(lab_test,lab_pred, labels=target_names))    
    #混淆矩陣
    target_names = ['0','1' ,'2','3','4','5']
    print(confusion_matrix(lab_test,lab_pred, labels=target_names))
    confusion = confusion_matrix(lab_test,lab_pred, labels=target_names)
    


def GRU_Model(shape):
    model = Sequential()
    model.add(GRU(units=32, input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=32, input_length=shape[1], input_dim=shape[2]))
    model.add(Dropout(0.2)) 
    # The output layer
    model.add(Dense(units=1))
    model.add(Activation('relu'))
    model.summary()
    return model


def BILSTM_Model(shape):
    from keras.layers import TimeDistributed
    from keras.layers import Bidirectional
    
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_length=shape[1], input_dim=shape[2], return_sequences=True), input_shape=(shape[1],shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, input_length=shape[1], input_dim=shape[2]), input_shape=(shape[1],shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('relu'))    
    
    model.summary()
    return model


#LSTM
def buildManyToOneModel(shape):   
    model = Sequential()
    model.add(LSTM(32, input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, input_length=shape[1], input_dim=shape[2]))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model



#級距分類_GBTR、SVM
def OthersModel_Score(Y_test,y_pred_test):
    #淹水級距分類
    lab_test = []
    lab_pred = []
    for i in range(len(Y_test)):
        if 0.1>=Y_test.values.tolist()[i]>=0:lab_test.append('0')
        elif 0.3>=Y_test.values.tolist()[i]>0.1:lab_test.append('1')
        elif 0.5>=Y_test.values.tolist()[i]>0.3:lab_test.append('2')
        elif 1>=Y_test.values.tolist()[i]>0.5:lab_test.append('3')
        elif 2>=Y_test.values.tolist()[i]>1:lab_test.append('4')
        elif Y_test.values.tolist()[i]>2:lab_test.append('5')
        else:lab_test.append('-1')
    for i in range(len(y_pred_test)):
        if 0.1>=y_pred_test[i]>=0:lab_pred.append('0')
        elif 0.3>=y_pred_test[i]>0.1:lab_pred.append('1')
        elif 0.5>=y_pred_test[i]>0.3:lab_pred.append('2')
        elif 1>=y_pred_test[i]>0.5:lab_pred.append('3')
        elif 2>=y_pred_test[i]>1:lab_pred.append('4')
        elif y_pred_test[i]>2:lab_pred.append('5')
        else:lab_pred.append('-1')    
    
    target_names = ['-1','0','1' ,'2','3','4','5']
    print('準確率：', accuracy_score(lab_test, lab_pred))
    print("report:\n",classification_report(lab_test,lab_pred, labels=target_names))
    
    #混淆矩陣
    target_names = ['0','1' ,'2','3','4','5']
    print(confusion_matrix(lab_test,lab_pred, labels=target_names))
    confusion = confusion_matrix(lab_test,lab_pred, labels=target_names)
    

#級距分類_LSTM、GRU、BiLSTM
def NN_Score(X_train, Y_train, X_test, Y_test):
    #用模型預測
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    #評估模型 - R平方
    print("The R2 score on the Train set is:\\t{:0.3f}".format(r2_score(Y_train, y_pred_train)))
    print("The R2 score on the Val set is:\\t{:0.3f}".format(r2_score(Y_val, y_pred_val)))
    print("The R2 score on the Test set is:\\t{:0.3f}".format(r2_score(Y_test, y_pred_test)))    
    #評估模型 - MAE
    print('MAE Train：\\t{:0.3f}'.format(mean_absolute_error(Y_train, y_pred_train)))
    print('MAE Val：\\t{:0.3f}'.format(mean_absolute_error(Y_val, y_pred_val)))
    print('MAE Test：\\t{:0.3f}'.format(mean_absolute_error(Y_test, y_pred_test)))    
    #評估模型 - RMSE
    print('RMSE Train：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_train, y_pred_train))))
    print('RMSE Val：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_val, y_pred_val))))
    print('RMSE Test：\\t{:0.3f}'.format(np.sqrt(mean_squared_error(Y_test, y_pred_test))))
    
    #淹水級距分類
    lab_test = []
    lab_pred = []
    for i in range(len(Y_test)):
        if 0.1>=Y_test[i]>=0:lab_test.append('0')
        elif 0.3>=Y_test[i]>0.1:lab_test.append('1')
        elif 0.5>=Y_test[i]>0.3:lab_test.append('2')
        elif 1>=Y_test[i]>0.5:lab_test.append('3')
        elif 2>=Y_test[i]>1:lab_test.append('4')
        elif Y_test[i]>2:lab_test.append('5')
        else:lab_test.append('-1')
    for i in range(len(y_pred_test)):
        if 0.1>=y_pred_test[i]>=0:lab_pred.append('0')
        elif 0.3>=y_pred_test[i]>0.1:lab_pred.append('1')
        elif 0.5>=y_pred_test[i]>0.3:lab_pred.append('2')
        elif 1>=y_pred_test[i]>0.5:lab_pred.append('3')
        elif 2>=y_pred_test[i]>1:lab_pred.append('4')
        elif y_pred_test[i]>2:lab_pred.append('5')
        else:lab_pred.append('-1')    
    
    target_names = ['-1','0','1' ,'2','3','4','5']
    print('準確率：', accuracy_score(lab_test, lab_pred))
    print("report:\n",classification_report(lab_test,lab_pred, labels=target_names))
    
    #混淆矩陣
    #target_names = ['0','1' ,'2','3','4','5']
    target_names = ['0','1' ,'2','3']
    print(confusion_matrix(lab_test,lab_pred, labels=target_names))
    confusion = confusion_matrix(lab_test,lab_pred, labels=target_names)
    
    

if __name__ == '__main__':
    
    '''''''''''''''''''''''''''''''''讀取原始資料'''''''''''''''''''''''''''''''''
    
    path_ground = "G:/Flooding/data/Ground" #要列出所有地面資料的目錄
    ignore_files = ["dm1d0000.asc","dm1maxd0.asc","SIMULATE.REP","sobek.log","dm1d0001.map","SOBEK.LOG"] #無須讀取的檔案
    split_gruond = []#切出第N塊網格地面淹水量
    
    path_rainfall = "G:/Flooding/data/Rainfall" #要列出所有地面資料的目錄
    split_rainfall = [] #切出第N塊網格雨量
        
    #Ground_批出讀取子目錄與檔案
    for root, dirs, files in walk(path_ground):
        files = [x for x in files if x not in ignore_files]
        for name in files:
            #print(root+"\\"+name)
            allroot = root+"/"+name
            try:
                Ground_Read(allroot)
            except:
                print("except",allroot)    
    
    #Rainfall_批出讀取子目錄與檔案
    for root, dirs, files in walk(path_rainfall):
        for name in files:
            #print(root+"\\"+name)
            allroot = root+"\\"+name
            try:
                Railfall_Read(allroot)
            except:
                print("except",allroot)
    
    
    data={"Ground":split_gruond,"Rainfall" : split_rainfall}#轉成字典
    data_df=DataFrame(data)#轉成DataFrame
    #print(data_df)
    data_df.to_csv("C:/Flooding/data/34x43_218364_final.csv",index=False,sep=',') #匯出CSV   
    


    '''''''''''''''''''''''''''''''''Model'''''''''''''''''''''''''''''''''
    
    ''''GBTR, SVM, ARIMA''''
    train_norm = pd.read_csv("C:/Flooding/data/34x43_218364_final.csv") #讀取CSV檔    
    #切分資料集
    X_train, X_test, Y_train, Y_test = train_test_split(train_norm['Rainfall'].to_numpy().reshape(-1, 1), train_norm['Ground'], test_size=0.3,random_state=28)#將數據集分為訓練集和驗證集
    
    #模型預測_GBTR
    GradientBoostingRegressor_Model(X_train, Y_train, X_test, Y_test)  
    
    #模型預測_SVM
    SVM_Model(X_train, Y_train, X_test, Y_test) 
    
    #模型預測_ARIMA
    autocorrelation_plot(train_norm['Ground'])    
    # 自相關圖
    plot_acf(data_diff)
    plt.title('ACF(Ground)')
    plt.show()
    # 偏自相關圖
    plot_pacf(data_diff)
    plt.title('PACF(Ground)')
    plt.show()
    ARIMA_Model(train_norm) 
    

    ''''GRU, BILSTM, LSTM''''
    train_norm = pd.read_csv("C:/Flooding/data/34x43_218364_final.csv") #讀取CSV檔    
    X_train, Y_train = buildTrain(train_norm, 24, 1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = splitData(X_train, Y_train, 0.3)
    
    #模型預測_GRU
    model = GRU_Model(X_train.shape)
    model.compile(optimizer='adam',loss='mean_squared_error') # Compiling
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train,Y_train,epochs=30,batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback]) # Fitting to the training set
    NN_Score(X_train, Y_train, X_test, Y_test)
        
    #模型預測_BILSTM
    model = BILSTM_Model(X_train.shape)
    model.compile(loss= 'mean_squared_error' , optimizer= 'adam' , metrics=[ 'acc' ])
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
    NN_Score(X_train, Y_train, X_test, Y_test)
    
    #模型預測_LSTM
    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
    NN_Score(X_train, Y_train, X_test, Y_test)




