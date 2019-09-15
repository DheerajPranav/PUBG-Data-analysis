# -*- coding: utf-8 -*-
"""
@author: Dheeraj & Aditya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import gc, sys
def reduceMemory(df):
   
    import numpy as np

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


def visualize(col_name, num_bin=10):
    
    title_name = col_name[0].upper() + col_name[1:]
    f, ax = plt.subplots()
    plt.xlabel(title_name)
    plt.ylabel('log Count')
    ax.set_yscale('log')
    df_train.hist(column=col_name,ax=ax,bins=num_bin)
    plt.title('Histogram - ' + title_name)
    tmp = df_train[col_name].value_counts().sort_values(ascending=False)

    print('Min value of ' + title_name + ' is: ',min(tmp.index))
    print('Max value of ' + title_name + ' is: ',max(tmp.index))
    
def BuildFeature(is_train=True):
    
    y = None
    test_idx = None
    
    if is_train: 
        print("Reading train.csv")
        df = pd.read_csv('G:\PUBG\PUBG/train_V2.csv')           
        df = df[df['maxPlace'] > 1]
    else:
        print("Reading test.csv")
        df = pd.read_csv('G:\PUBG\PUBG/test_V2.csv')
        test_idx = df.Id
    
    # Reduce the memory usage
    df = reduceMemory(df)
    
    print("Delete Unuseful Columns")
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")  
    
    if is_train: 
        print("Read Labels")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("Read Group mean features")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    if is_train:
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId','groupId']]
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    print("Read Group max features")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("Read Group min features")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("Read Group size features")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("Read Match mean features")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    print("Read Match size features")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    X = df_out
    feature_names = list(df_out.columns)
    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx
develop_mode = False
if develop_mode:
    df_train = reduceMemory(pd.read_csv('G:\PUBG\PUBG/train_V2.csv', nrows=5000))
    df_test = reduceMemory(pd.read_csv('G:\PUBG\PUBG/test_V2.csv'))
else:
    df_train = reduceMemory(pd.read_csv('G:\PUBG\PUBG/train_V2.csv'))
    df_test = reduceMemory(pd.read_csv('G:\PUBG\PUBG/test_V2.csv'))
print('The sizes of the datasets are:')
print('Training Dataset: ', df_train.shape)
print('Testing Dataset: ', df_test.shape)

group_tmp = df_train[df_train['matchId']=='df014fbee741c6']['groupId'].value_counts().sort_values(ascending=False)

plt.figure()
plt.bar(group_tmp.index,group_tmp.values)
plt.xlabel('GroupId')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.title('Number of Group Members in One Match')
plt.show()


print('Min number of group members is: ',min(group_tmp.values))
print('Max number of group members is: ',max(group_tmp.values))

visualize('assists')
def MissValueAnalysis():
    miss_total = df_train.isnull().sum().sort_values(ascending=False)
    miss_percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([miss_total, miss_percent], axis=1, keys=['total', 'percent'])

    percent_data = miss_percent.head(20)
    percent_data.plot(kind="bar")
    plt.xlabel("Columns")
    plt.ylabel("Percentage")
    plt.title("Total Missing Value (%) in Training Data")
    plt.show()

    miss_total = df_test.isnull().sum().sort_values(ascending=False)
    miss_percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([miss_total, miss_percent], axis=1, keys=['total', 'percent'])

    percent_data = miss_percent.head(20)
    percent_data.plot(kind="bar")
    plt.xlabel("Columns")
    plt.ylabel("Percentage")
    plt.title("Total Missing Value (%) in Training Data")
    plt.show()

print(MissValueAnalysis())

def CorrelationAnalysis():
    corr = df_train.corr()
    f, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(corr,cbar=True, annot=True, 
                          square=True, fmt='.2f', 
                          cmap='YlGnBu')

print(CorrelationAnalysis())

df_train.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6), title='longestKill vs winPlacePerc')

def HealsVSwinPlacePerc():
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)
    plt.title('heals vs winPlacePerc box plot')
    fig.axis(ymin=0, ymax=1)

HealsVSwinPlacePerc()


