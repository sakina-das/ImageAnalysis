# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:37:31 2016

@author: team5
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import xml.etree.ElementTree as ET
import argparse
import re
import sys
import os
from sklearn.metrics import accuracy_score


def filterDataframe(columnname, df, lower_threshold, upper_threshold):
    df_filtered = pd.DataFrame({'count' : df.groupby([columnname]).size()})
    df_filtered=df_filtered.reset_index()
    df_filtered = df_filtered[(df_filtered["count"]>lower_threshold)& (df_filtered["count"]<upper_threshold)]
    print("Filtering using " + columnname + " give : " + str(len(df_filtered.index)) + " unique values")
    df = pd.merge(df, df_filtered, how='inner', on=columnname)
    return df
    
    
def computeUserLikeMatrix(df1, df2, indexValue, columnValue):
    df1["value"] = pd.Series(1,index=df1.index, dtype="category")
    df1=df1.pivot(index=indexValue, columns=columnValue, values="value")
    df1.replace(float("nan"),0,inplace=True)
    df1=df1.reset_index()
    df1 = pd.merge(df1, df2, how="inner", on="userid")
    return df1
        
def performLogisticRegression(df_train,df_test,dependentVariable):
    X_train = np.array(df_train.drop(["age","userid","gender"],1))
    y_train = np.array(df_train[dependentVariable])
    X_test = np.array(df_test.drop(["age","userid","gender"],1))  
    model = LogisticRegression(penalty='l2', C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    df_test["gender"] = y_pred

def performLogisticRegressionImage(df_train_oxford,df_test_oxford,key_var,model):
    
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
    performLogisticRegressionImage(trainData,testData,"gender", model)
    
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
        
        
def readTranscripts(df, textFilesPath):
    df['transcripts'] = pd.Series('', index=df.index)
    userid_array = df['userid']
    
    for i in range(0, len(df)):
        file = open(textFilesPath + userid_array[i] + ".txt", 'r')
        df.loc[i,'transcripts'] = file.read()
        file.close()
        return df
                    
def saveTranscriptInArray(df,textFilesPath):
    userid = []
    userid = df['userid']
    transcript = []
    for i in userid:
        f = open(textFilesPath+i+".txt", 'r')
        transdata = f.read()
        transcript.append(transdata +',')
        f.close()
        return transcript
                            
                            
def predictAge(df_test_liwc):
    count_vect_age = CountVectorizer( input=u'content',encoding=u'latin-1',decode_error=u'strict',strip_accents=None,   lowercase=True,preprocessor=None,tokenizer=None,stop_words=None,ngram_range=(1, 2),analyzer=u'word',max_df=1.0,min_df=0,max_features=None,vocabulary=None,binary=False,dtype=np.int64)
    X_train = count_vect_age.fit_transform(df_profiles['transcripts'])
    z_train = df_profiles['age']
    clf_age = MultinomialNB()
    clf_age.fit(X_train, z_train)
    df_test_liwc['age'] = pd.Series(0, index=df_test_liwc.index)
    X_test = count_vect_age.transform(df_test_liwc['transcripts'])
    z_pred = clf_age.predict(X_test)
    df_test_liwc['age'] = z_pred
    return df_test_liwc
                                
                                
def predictTraits(df_test_liwc, df_train):
    big5 = ['neu','ext','con','ope','agr']
    feature_list = [x for x in df_train.columns.tolist()[:] if not x in big5]
    feature_list.remove('userid')
    feature_list.remove('unnamed: 0')
    feature_list.remove('age')
    feature_list.remove('gender')
    for trait in big5:
        X_train = df_train[feature_list]
        y_train = df_train[trait]
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        df_test_liwc[trait] = pd.Series(0, 
             index=df_test_liwc.index)                        
        X_test = df_test_liwc[feature_list]
        y_pred = regr.predict(X_test)
        if(not(trait=='neu')):
            df_test_liwc[trait] = y_pred
        feature_list.append(trait)
    return df_test_liwc
                                            
                                            
def predictNeu(df_test_liwc, df_train):
    big5 = ['neu']
    
    feature_list = [x for x in df_train.columns.tolist()[:] if not x in big5]
    feature_list.remove('userid')
    feature_list.remove('unnamed: 0')
    feature_list.remove('age')
    feature_list.remove('gender') 
    for trait in big5:
        X_train = df_train[feature_list]
        y_train = df_train[trait]
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        X_test = df_test_liwc[feature_list]
        y_pred = regr.predict(X_test)
        df_test_liwc[trait] = y_pred  
    return df_test_liwc
                                                    
                                                    
def predictGender_NB(df_profiles, df_t):
    count_vect = CountVectorizer( input=u'content',encoding=u'latin-1',decode_error=u'strict',strip_accents=None,   lowercase=True,preprocessor=None,tokenizer=None,stop_words=None,ngram_range=(1, 2),analyzer=u'word',max_df=1.0,min_df=0,max_features=None,vocabulary=None,binary=False,dtype=np.int64)
    X_train = count_vect.fit_transform(df_profiles['transcripts'])
    y_train = df_profiles['gender']
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    X_test = count_vect.transform(df_t['transcripts'])
    y_predicted = clf.predict(X_test)
    df_t['gender'] = y_predicted
    df_t['gender'].replace([1,0],['Female','Male'],inplace=True)
    return df_t


def predictAge_NB(df_profiles, df_t, text,tokens):
    count_vect = CountVectorizer( input=u'content',encoding=u'latin-1',decode_error=u'strict',strip_accents=None,   lowercase=True,preprocessor=None,tokenizer=customTokenizer,stop_words=None,ngram_range=(1, 2),analyzer=u'word',max_df=10.0,min_df=5,max_features=None,vocabulary=list(set(text)),binary=False,dtype=np.int64)
    X_train = count_vect.fit_transform(df_profiles['transcripts'])
    y_train = np.asarray(df_profiles['age'],dtype=np.int64)
    # z_train = df_profiles['gender']
    clf = MultinomialNB()
    clf2 = MultinomialNB()
    clf.fit(X_train,y_train)
    #clf2.fit(X_train,z_train)
    X_test = count_vect.transform(df_t['transcripts'])
    y_predicted = clf.predict(X_test)
    #z_predicted = clf2.predict(X_test)
    df_t['age'] = y_predicted
    #df_t['gender'] = z_predicted
    #df_t['gender'].replace([1,0],['Female','Male'],inplace=True)
    return df_t
                                                            
def createXMLFiles(df_test_liwc):
    #create xml files for each user
    for number in range(0, len(df_test_liwc)):
        if df_test_liwc.loc[number,'age'] < 25:
            age_xml = 'xx-24'
        elif df_test_liwc.loc[number,'age'] < 35:
            age_xml = '25-34'
        elif df_test_liwc.loc[number,'age'] <50:
            age_xml = '35-49'
        else:
            age_xml = '50-xx'
            root = ET.Element("user", age_group = age_xml, id=df_test_liwc.loc[number,'userid'], open = str(df_test_liwc.loc[number,'ope']),
            extrovert=str(df_test_liwc.loc[number,'ext']), neurotic=str(df_test_liwc.loc[number,'neu']),
            agreeable=str(df_test_liwc.loc[number,'agr']), conscientious=str(df_test_liwc.loc[number,'con']),
            gender = df_test_liwc.loc[number,'gender'])
            varUser = df_test_liwc.loc[number,'userid']
            tree = ET.ElementTree(root)
            tree.write(args.output_filepath + varUser + ".xml")
            
            emoticon_string = r"""
            (?:
            [<>]?
            [:;=8]                     # eyes
            [\-o\*\']?                 # optional nose
            [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
            |
            [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
            [\-o\*\']?                 # optional nose
            [:;=8]                     # eyes
            [<>]?
            )"""
            
            # The components of the tokenizer:
            regex_strings = (
            # Phone numbers:
            r"""
            (?:
            (?:            # (international)
            \+?[01]
            [\-\s.]*
            )?            
            (?:            # (area code)
            [\(]?
            \d{3}
            [\-\s.\)]*
            )?    
            \d{3}          # exchange
            [\-\s.]*   
            \d{4}          # base
            )"""
            ,
            # Emoticons:
            emoticon_string
            ,    
            # HTML tags:
            r"""<[^>]+>"""
            ,
            # Twitter username:
            r"""(?:@[\w_]+)"""
            ,
            # Twitter hashtags:
            r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
            ,
            # Remaining word types:
            r"""
            (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
            |
            (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
            |
            (?:[\w_]+)                     # Words without apostrophes or dashes.
            |
            (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
            |
            (?:\S)                         # Everything else that isn't whitespace.
            """
            )
            
            
            # This is the core tokenizing regex:
            
            word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)
            
            # The emoticon string gets its own regex so that we can preserve case for them as needed:
            emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)
            
            # These are for regularizing HTML entities to Unicode:
            html_entity_digit_re = re.compile(r"&#\d+;")
            html_entity_alpha_re = re.compile(r"&\w+;")
            amp = "&amp;"
                                                                        
class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case
        
    def tokenize(self,s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """   
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
            # Fix HTML character entitites:
            s = self.__html2unicode(s)
            # Tokenize:
            words = word_re.findall(s)
            # Possible alter the case, but avoid changing emoticons like :D into :d:
            if not self.preserve_case:            
                words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
            return words
                
                                                                                            
                                                                                            
def __html2unicode(self, s):
    """
    Internal metod that seeks to replace all the HTML entities in
    s with their corresponding unicode characters.
    """
    # First the digits:
    ents = set(html_entity_digit_re.findall(s))
    if len(ents) > 0:
        for ent in ents:
            entnum = ent[2:-1]
            try:
                entnum = int(entnum)
                s = s.replace(ent, unichr(entnum))	
            except:
                pass
                # Now the alpha versions:
                ents = set(html_entity_alpha_re.findall(s))
                ents = filter((lambda x : x != amp), ents)
                for ent in ents:
                    entname = ent[1:-1]
                    try:            
                        s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
                    except:
                        pass                    
                        s = s.replace(amp, " and ")
                return s
                        
                        
def forToken(transcript):
    text = ""
    #if __name__ == '__main__':
    tok = Tokenizer(preserve_case=False)
    for s in transcript:
        text += unicode(s,errors='replace')
        
        tokens = tok.tokenize(text)
        return text,tokens
        ###I am checking the tokenizer here

def customTokenizer(tokens):
    return[i for i in tokens]


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="--input file path", dest="input_filepath")
parser.add_argument("-o", help="--output file path", dest="output_filepath")
args = parser.parse_args()
LIWCTestFilePath = args.input_filepath + "LIWC/LIWC.csv"
relationsFilePath = args.input_filepath + "relation/relation.csv"
profileFilePath = args.input_filepath + "profile/profile.csv"
inputTextFilePath = args.input_filepath + "text/"

#
#LIWCTestFilePath = os.path.abspath("LIWC/LIWC.csv")
#relationsFilePath = os.path.abspath("relation/relation.csv")
#profileFilePath = os.path.abspath("profile/profile.csv")
#inputTextFilePath = os.path.abspath("text/")

#
#LIWCTestFilePath = (r"D:\MCSS\MachineLearning\Project_User_Profiling\LIWCfeatures\public-test\LIWC\LIWC.csv")
#relationsFilePath = (r"D:\MCSS\MachineLearning\Project_User_Profiling\data\public-test-data\relation\relation.csv")
#profileFilePath = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\profile\profile.csv")
#inputTextFilePath = (r"D:\MCSS\MachineLearning\Project_User_Profiling\data\public-test-data\text")
#
#oxford_train = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\training\oxford.csv")
#profile_train = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\training\profile\profile.csv")
#profile_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\profile\profile.csv")
#oxford_test = (r"D:\MCSS\MachineLearning\Project_User_Profiling\tcss555\public-test-data\oxford.csv")


oxford_train = "/data/training/oxford.csv"
profile_train = "/data/training/profile/profile.csv"
profile_test = args.input_filepath + "profile/profile.csv"
oxford_test = args.input_filepath + "oxford.csv"


#import data for training
profileTrainPath = "/data/training/profile/profile.csv"
liwcTrainPath = "/data/training/LIWC/LIWC.csv"
relationsTrainPath = "/data/training/relation/relation.csv"
textTrainPath = '/data/training/text/'

print("Please wait... Processing")
"""profileTrainPath = "D:/Anurita/UW/Fall 2016/ML/Week3/training/profile/profile.csv"
liwcTrainPath = "D:/Anurita/UW/Fall 2016/ML/Week3/training/LIWC/LIWC.csv"
relationsTrainPath = "D:/Anurita/UW/Fall 2016/ML/Week3/training/relation/relation.csv"
textTrainPath = "D:/Anurita/UW/Fall 2016/ML/Week3/training/text/"
"""


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
    performLogisticRegressionImage(trainData,testData,"age", model)

        
df_train_oxford = pd.read_csv(oxford_train)
df_train_profiles = pd.read_csv(profile_train)
#
df_test_profiles = pd.read_csv(profile_test)
df_test_oxford = pd.read_csv(oxford_test)

df_train_profiles.columns = [x.lower() for x in df_train_profiles.columns]
df_train_oxford.columns = [x.lower() for x in df_train_oxford.columns]
df_test_profiles.columns = [x.lower() for x in df_test_profiles.columns]
df_test_oxford.columns = [x.lower() for x in df_test_oxford.columns]

#
#GenderProfiling(df_train_oxford.copy(deep=True), df_train_profiles.copy(deep=True))
#AgeGroupProfiling(df_train_oxford, df_train_profiles)

GenderProfiling(df_train_oxford.copy(deep=True), df_test_oxford.copy(deep=True), df_train_profiles.copy(deep=True), df_test_profiles.copy(deep=True))
AgeGroupProfiling(df_train_oxford, df_test_oxford, df_train_profiles, df_test_profiles)

print("Above: Gender and age prediction accuracy from images") 

df_profiles = pd.read_csv(profileTrainPath)
df_liwc = pd.read_csv(liwcTrainPath)

df_relations_train = pd.read_csv(relationsTrainPath)
df_profile_train = pd.read_csv(profileTrainPath)
df_profiles.columns = [x.lower() for x in df_profiles.columns]
df_liwc.columns = [x.lower() for x in df_liwc.columns]
df_profile_train.drop(["ope", "neu","con","ext","Unnamed: 0","agr"],axis=1,inplace=True)
df_relations_train.columns = [x.lower() for x in df_relations_train.columns]
df_profile_train.columns = [x.lower() for x in df_profile_train.columns]
df_relations_train = pd.merge(df_profile_train, df_relations_train, how='inner', on='userid')
df_relations_train['gender'].replace([1,0],['Female','Male'],inplace=True)
#filter merge data with userid and likeid
df_relations_train=filterDataframe("userid", df_relations_train, 10, 1500)
df_relations_train=filterDataframe("like_id", df_relations_train, 5, 400)
uniqueLikesTrain = df_relations_train.like_id.unique()

df_test_profile = pd.read_csv(profileFilePath)
df_test_relations = pd.read_csv(relationsFilePath)

df_test_profile.columns = [x.lower() for x in df_test_profile.columns]
df_test_profile.drop(["ope", "neu","con","ext","unnamed: 0","agr"],axis=1,inplace=True)
df_test_relations.columns = [x.lower() for x in df_test_relations.columns]
df_test_relations.drop(["unnamed: 0"], axis=1, inplace=True)
df_test_relations = pd.merge(df_test_profile, df_test_relations, how='inner', on='userid')
#df_test_relations = filterDataframe("like_id",df_test,1)

uniqueLikesTest = df_test_relations.like_id.unique()
commonLikes = pd.Series(list(set(uniqueLikesTrain).intersection(set(uniqueLikesTest))))
df_test_relations = df_test_relations[df_test_relations['like_id'].isin(commonLikes)]
df_relations_train = df_relations_train[df_relations_train['like_id'].isin(commonLikes)]

df_relations_train =computeUserLikeMatrix(df_relations_train, df_profile_train, "userid","like_id")
df_test_relations =computeUserLikeMatrix(df_test_relations, df_test_profile, "userid","like_id")
print("Predicting gender")
performLogisticRegression(df_relations_train,df_test_relations,"gender")
df_test_relations['gender'].replace([1,0],['Female','Male'],inplace=True)
del(df_relations_train)
del(df_test_profile)
del(df_profile_train)

df_gender = df_test_relations[["userid", "gender"]]
#merge the two datasets into single dataframe using userid                                
df_train = pd.merge(df_profiles, df_liwc, how='inner', on='userid')

#read test data for liwc features
df_test_liwc = pd.read_csv(LIWCTestFilePath)
df_test_liwc.columns = [x.lower() for x in df_test_liwc.columns]
print("Predicting traits")
df_test_liwc = predictTraits(df_test_liwc, df_train)
df_test_liwc = predictNeu(df_test_liwc, df_train)
#read transcripts and store it against user from text file
df_profiles = readTranscripts(df_profiles, textTrainPath)


transcript = saveTranscriptInArray(df_profiles,textTrainPath) #############################update
text,tokens = forToken(transcript) #########updated


df_test_liwc = readTranscripts(df_test_liwc, inputTextFilePath)
print("Predicting age")



#############################check here#############################
#df_age_NB = predictAge_NB(df_profiles, df_age_NB,text)
#df_test_liwc = predictAge_NB(df_profiles, df_test_liwc,text,tokens)
df_test_liwc = predictAge(df_test_liwc)
if(len(df_test_liwc.index) > len(df_gender.index)):
    print("Gender prediction using Naive Bayes for users without likeids")
    # df_test_liwc = pd.merge(df_gender, df_test_liwc, how="outer", on="userid")
    df_gender_NB = df_test_liwc[~df_test_liwc['userid'].isin(df_gender['userid'])]
    df_gender_NB = predictGender_NB(df_profiles, df_gender_NB)  
    column_list = ["userid", "gender"]
    df_gender_NB = df_gender_NB[column_list]
    frames = [df_gender_NB, df_gender]
    df_gender = pd.concat(frames)
    
    
    df_test_liwc = pd.merge(df_gender, df_test_liwc, how="inner", on="userid")
    print("Creating XMLs")
    createXMLFiles(df_test_liwc)
