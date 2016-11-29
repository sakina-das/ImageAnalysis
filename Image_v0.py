# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:55:52 2016

@author: Sakina:ImageAnalysis
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score


oxford_train = ("/data/training/oxford.csv")
profile_train = ("/data/training/profile/profile.csv")
#profile_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\profile\profile.csv")
#oxford_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\oxford.csv")
##
#oxford_train = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\training\oxford.csv")
#profile_train = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\training\profile\profile.csv")
#profile_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\profile\profile.csv")
#oxford_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\oxford.csv")

def performLogisticRegression(df_train_oxford,df_test_oxford,key_var,model):
    
    train_X = np.array(df_train_oxford.drop(["userid", "faceid", key_var],1))
    train_y = df_train_oxford[key_var]

    model.fit(train_X, 
              train_y)

    test_X = np.array(df_test_oxford.drop(["userid", "faceid", key_var],1))
    y_pred = model.predict(test_X)    
    validation(df_train_oxford, model, key_var)


def validation(df_train_oxford, model, key_var):
    [train, test] = split_data(df_train_oxford, train_perc = 0.8)
    
    train_X = np.array(train.drop(["userid", "faceid", key_var],1))
    train_y = train[key_var]
    model.fit(train_X, 
                  train_y)
    
    test_X = np.array(test.drop(["userid", "faceid", key_var],1))
    y_pred = model.predict(test_X)
    score = accuracy_score(test[key_var], y_pred)
    print("Accuracy for ", key_var, " profiling: ", score)
   
def split_data(df, train_perc = 0.8):

    df['train'] = np.random.rand(len(df)) < train_perc
     
    train = df[df.train == 1]
    test = df[df.train == 0]
    train = train.drop(['train'], 1)
    test = test.drop(['train'], 1)
    return [train, test]
 
def GenderProfiling(df_train_oxford, df_test_oxford, df_train_profiles, df_test_profiles):
    trainProfiles = df_train_profiles
    testProfiles = df_test_profiles
    
    trainProfiles.drop(["ope", "unnamed: 0", "age", "con", "ext", "agr", "neu"], axis = 1, inplace = True)
    trainData = pd.merge(trainProfiles, df_train_oxford, how='inner', on='userid')
    
    testProfiles.drop(["ope", "unnamed: 0", "age", "con", "ext", "agr", "neu"], axis = 1, inplace = True)
    testData = pd.merge(testProfiles, df_test_oxford, how='inner', on='userid')

    model = linear_model.LogisticRegression()
    performLogisticRegression(trainData,testData,"gender", model)

def ComputeAgeGroup(item):
    if (item < 10):
        return 0
    elif (item <25):
        return 1
    elif (item <45):
        return 2
    elif (item < 67):
        return 3
    else:
        return 4
    
def AgeGroupProfiling(df_train_oxford, df_test_oxford, df_train_profiles, df_test_profiles):
    trainProfiles = df_train_profiles
    testProfiles = df_test_profiles
    
    trainProfiles['age'] = trainProfiles['age'].apply(ComputeAgeGroup)
    trainProfiles.drop(["ope", "unnamed: 0", "con", "ext", "agr", "neu", "gender"], axis = 1, inplace = True)
    trainData = pd.merge(trainProfiles, df_train_oxford, how='inner', on='userid')
    
   # trainData.drop(['age'], 1, inplace=True)


    #testProfiles['age'].apply(lambda x: ComputeAgeGroup(x)) -> will be predicted
    testProfiles.drop(["ope", "unnamed: 0", "con", "ext", "agr", "neu", "gender"], axis = 1, inplace = True)
    testData = pd.merge(testProfiles, df_test_oxford, how='inner', on='userid')

    #testData.drop(['age'], 1, inplace=True)
    
    model = linear_model.LogisticRegression()
    performLogisticRegression(trainData,testData,"age", model)

        
df_train_oxford = pd.read_csv(oxford_train)
df_train_profiles = pd.read_csv(profile_train)
#
#df_test_profiles = pd.read_csv(profile_test)
#df_test_oxford = pd.read_csv(oxford_test)

df_train_profiles.columns = [x.lower() for x in df_train_profiles.columns]
df_train_oxford.columns = [x.lower() for x in df_train_oxford.columns]
#df_test_profiles.columns = [x.lower() for x in df_test_profiles.columns]
#df_test_oxford.columns = [x.lower() for x in df_test_oxford.columns]


GenderProfiling(df_train_oxford.copy(deep=True), df_train_profiles.copy(deep=True))
AgeGroupProfiling(df_train_oxford, df_train_profiles)

#GenderProfiling(df_train_oxford.copy(deep=True), df_test_oxford.copy(deep=True), df_train_profiles.copy(deep=True), df_test_profiles.copy(deep=True))
#AgeGroupProfiling(df_train_oxford, df_test_oxford, df_train_profiles, df_test_profiles)

print("End of Program.") 