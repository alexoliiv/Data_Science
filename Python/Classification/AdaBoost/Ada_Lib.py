
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd 
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import sklearn.metrics
from scipy import stats


#########################################################################################
#                       Data Manipulation Functions                                    #
####################################################################################### 

def Import_Data(path):
    '''
    path = path containing the dataset location
    Function : Import_Data()
    Date : 08/05/2019
    Author : Alexander Leite
    Import "path" dataset and print a few informations about it.
    '''
    Df = pd.read_csv(path,sep= ',')
     # Printing the dataset shape 
    print ("Dataset Lenght: ", len(Df)) 
    print ("Dataset Shape: ", Df.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset:\n ",Df.head()) 
    return Df

def Split_dataset(df,corr):
    '''
    df = Dataset 
    Function : Split_dataset()
    Date : 08/05/2019
    split the dataset into train and test instances
    '''
    if pd.core.series.Series == type(corr):
        X = df.drop(['target']+list(corr.index),axis=1)
    else:
        X = df.drop(['target'],axis=1)
    y = df['target']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)
    print("X train shape : ", X_train.shape)
    print("y train shape : ", y_train.shape)
    print("X test  shape : ", X_test.shape)
    print("y test  shape : ", y_test.shape)
    return X_train,X_test,y_train,y_test





#########################################################################################
#                       Data Visualization/EDA Functions                               #
####################################################################################### 

def Data_Summaryze(df):
    '''
    df = Dataset 
    Function : Data_Summaryze()
    Date : 08/05/2019
    Summaryze a few "df" informations like : null values, data types and some descriptive measures.
    '''
    ## Visualyze data types by columns
    print("Data type :\n",df.dtypes,'\n')
    fig = plt.figure(figsize=(10,10))
    ## Axis 1
    ax1 = fig.add_subplot(2,2,1)
    ax1.bar(df.columns,df.isnull().sum())
    ax1.set_ylim(0,len(df))
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Quantity')
    plt.xticks(rotation='vertical')
    plt.title('Null Values',fontsize=18)
    ## Axis 2
    ax2 = fig.add_subplot(2,2,2)
    ax2.bar(['Has Heart Disease','Doesn\'t have heart disease'],df.target.value_counts())
    plt.title('Categories',fontsize=18)
    plt.xticks(rotation='vertical')
    return

def Identify_outliers(df):
    '''
    X = X feature matrix
    clf_object = our model object
    Function : Identify_outliers()
    Date : 22/05/2019
    identify outliers
    Return : outliers
    '''
    ## Get Z-Scores for whole dataset
    z_scores = np.abs(stats.zscore(df))
    row = np.where(z_scores > 3)[0]
    column = np.where(z_scores > 3)[1]
    index = list()
    for row, column in zip(np.where(z_scores > 3)[0],np.where(z_scores > 3)[1]):
        index.append([row,column])
        
    return index

def Correlation_Matrix(df):
    '''
    df = Dataset 
    Function : Correlation_Matrix()
    Date : 08/05/2019
    plot a correlation matrix
    '''
    plt.figure(figsize=(12,10))
    corr = df.corr()
    corr = corr[(abs(corr > 0.3) | abs(corr < -0.3))]['target'].dropna()[:-1]
    print('These',len(corr),'variables are correlated with our target:\n',corr)
    sns.heatmap(df.corr(),cmap='viridis',vmax=1.0,vmin=-1.0,annot=True,annot_kws={"size":8},square=True)
    return corr

def Plot_Scores(df):
    '''
    df = Dataset containing the model scores
    Function : Correlation_Matrix()
    Date : 08/05/2019
    plot a graph containing the model scores
    '''
    
    # create plot
    fig = plt.figure(figsize=(12,4))
    index = np.arange(df.shape[0])
    bar_width = 0.35
    opacity = 0.8
    
    index1 = np.arange(len(index))
    index2 = [x + bar_width for x in index1]
    index3 = [x + bar_width for x in index2]
    
    rects1 = plt.bar(index1, df['test_accuracy'], 0.25,edgecolor='white',
    alpha=opacity,
    color='#7f6d5f',
    label='Test Accuracy')
    
    rects2 = plt.bar(index2, df['test_precision'], 0.25,edgecolor='white',
    alpha=opacity,
    color='#557f2d',
    label='Test Precision')
    
    rects3 = plt.bar(index3, df['test_recall'], 0.25,edgecolor='white',
    alpha=opacity,
    color='#2d7f5e',
    label='Test Recall')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    plt.title('Scores by algorithm')
    plt.xticks(index2 , ('Decision Tree', 'Random Forest', 'AdaBoost'))
    plt.ylim((0,1))
    plt.legend()
    
    plt.autoscale(tight=True,axis=0)
    plt.show()
    
    return

#########################################################################################
#                                  Modelling Functions                                 #
####################################################################################### 

def AdaBoost_Fit(X_train,y_train,base_estimator):
    '''
    X = features
    y = target
    Function : AdaBoost_Classifier()
    Date : 09/05/2019
    create our adaboost model to fit and predict values for the dataset
    '''
    # Create adaboost classifer object
    Abc = AdaBoostClassifier(base_estimator=base_estimator,random_state=123)
    Abc.fit(X_train,y_train)
    return Abc


def AdaBoost_Predict(X_test,clf):
    '''
    X = features
    y = target
    Function : AdaBoost_Classifier()
    Date : 09/05/2019
    create our adaboost model to fit and predict values for the dataset
    '''
    y_pred = clf.predict(X_test)
    return y_pred

def Create_Random_Grid(clf):
    '''
    Function : Randomize_Grid()
    Date : 15/04/2019
    Create a dictionary with our parameter grid.
    Return : dict with parameters
    '''
    ## Hyperparameter Tuning
    # number of weak learners
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 150, num = 15)]
    # learning rate
    learning_rate = [x for x in np.linspace(start = 0.5, stop = 1.5, num = 10)]
    # boosting algorithm
    algorithm = ['SAMME','SAMME.R']
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate,
                   'algorithm': algorithm,
                   }        
            
        
    return random_grid   

def Random_Grid_Fit(clf,random_grid,X_train,y_train):
    '''
    clf : model object
    random_grid : our random grid parameters object
    X_train : X matrix
    y_train : y vector
    Function : Random_Grid_Fit()
    Date : 15/04/2019
    Fit our random forest with our random grid parameteres
    Return : The "best" params found by our random search
    '''
    clf_random = RandomizedSearchCV(estimator=clf,param_distributions=random_grid, n_iter=100, cv = 30,
    verbose = 2, n_jobs=-1, iid = False)
    
    clf_random.fit(X_train,y_train)
    # plot
    ##fig, ax = plt.subplots()
    ##ax.plot(k_range,scores_list)
    ##ax.grid()
    ##print(range_list)
    return clf_random.best_params_,clf_random.best_estimator_

def Random_Tree_Fit(X,y):
    '''
    X = X feature matrix
    y = y target vector
    Function : Random_Tree_Fit()
    Date : 15/04/2019
    Create our random forest model and fit X and y to the model, so, we can make our predictions later.
    Return : "df" splitted in "X_train","X_test","y_train","y_test"
    '''
    ## Model fitting
    clf = RandomForestClassifier(min_samples_split=30, min_samples_leaf=10,random_state=0)
    clf.fit(X,y)
    return clf

def Random_Grid_Predict(X,clf_object):
    '''
    X = X feature matrix
    clf_object = our model object
    Function : Random_Tree_Predict()
    Date : 15/04/2019
    Predict values with our "X_train" matrix, to calculate accuracy later.
    Return : our predicted values.
    '''
    ## Model Prediction
    predictions = clf_object.predict(X)
    return predictions

def KFold_Validation(X,y,lplot,base_estimator):
    '''
    X = X feature matrix
    clf_object = our model object
    Function : AdaBoost_Classifier_KFold()
    Date : 15/05/2019
    Predict values with our "X_train" matrix, to calculate accuracy later.s
    Return : our predicted values.
    '''
    scoring = ('accuracy','precision','recall')
    classifier = str(base_estimator)
    scores_dict = dict()
    score_mean = list()
    Abc = AdaBoostClassifier(base_estimator=base_estimator)
    scores_dict = cross_validate(Abc,X,y,cv=10,scoring=scoring,return_train_score=False)
    # plot
    if lplot:
        fig, ax = plt.subplots()
        ax.plot(k_range,scores_list)
        ax.grid()
    ##print(range_list)
    nParenthesis = classifier.find('(')
    score_mean.append([scores_dict.get('test_accuracy').mean(),scores_dict.get('test_precision').mean(),scores_dict.get('test_recall').mean(),classifier[0:nParenthesis]])
    return score_mean

#########################################################################################
#                          Model Evalatuaion Functions                                 #
####################################################################################### 

def Calc_Accuracy(y_test, y_pred):
    print("Confusion Matrix: ", 
    confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print(classification_report(y_test, y_pred)) 
    return
