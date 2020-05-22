import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def RemoveOutliersZScore(dataset,column):
    dataset = dataset[abs(stats.zscore(dataset[column]))<=3]
    return dataset

def CountOutliersZScore(dataset,column):
    return len(dataset[abs(stats.zscore(dataset[i]))>3])

df = pd.read_csv("Training_Dataset_v2.csv")

Col_Rename = {"sku":"Product ID",
"national_inv":"Current inventory level",
"lead_time":"Product Transit Time",
"in_transit_qty":"Transit Product Amount",
"forecast_3_month":"Forecast sales 3 months",
"forecast_6_month":"Forecast sales 6 months",
"forecast_9_month":"Forecast sales 9 months",
"sales_1_month":"Sales prior 1 month",
"sales_3_month":"Sales prior 3 month",
"sales_6_month":"Sales prior 6 month",
"sales_9_month":"Sales prior 9 month",
"min_bank":"amount to stock",
"potential_issue":"Source issue identified",
"pieces_past_due":"Parts overdue from source",
"perf_6_month_avg":"Source performance prior 6 month",
"perf_12_month_avg":"Source performance prior 12 month",
"local_bo_qty":"Amount of stock orders overdue",
"deck_risk":"Part risk flag1",
"oe_constraint":"Part risk flag2",
"ppap_risk":"Part risk flag3",
"stop_auto_buy":"Part risk flag4",
"rev_stop":"Part risk flag5",
"went_on_backorder":"Product Backordered"}


df.rename(columns=Col_Rename,inplace=True)

df.loc[1687860,"Product ID"] = "1687860"

df["Product ID"] = df["Product ID"].astype("int")

df[["Source issue identified","Part risk flag1","Part risk flag2","Part risk flag3","Part risk flag4","Part risk flag5","Product Backordered"]] = df[["Source issue identified","Part risk flag1","Part risk flag2","Part risk flag3","Part risk flag4","Part risk flag5","Product Backordered"]].astype("category")

df.drop(index=1687860,inplace=True)

df.loc[df[df["Product Transit Time"].isnull()][df[df["Product Transit Time"].isnull()]["Transit Product Amount"]==0].index,"Product Transit Time"]=0

df.loc[df["Product Transit Time"].isnull(),"Product Transit Time"]=df["Product Transit Time"].median()

df_new = df.copy()

sums = 0
for i in ['Product ID', 'Current inventory level', 'Product Transit Time',
       'Transit Product Amount', 'Forecast sales 3 months',
       'Forecast sales 6 months', 'Forecast sales 9 months',
       'Sales prior 1 month', 'Sales prior 3 month', 'Sales prior 6 month',
       'Sales prior 9 month', 'amount to stock', 'Parts overdue from source',
       'Source performance prior 6 month', 'Source performance prior 12 month',
       'Amount of stock orders overdue']:
    df_new = RemoveOutliersZScore(df_new,i)

df_new["Product Backordered"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Product Backordered"]))
df_new["Source issue identified"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Source issue identified"]))
df_new["Part risk flag1"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Part risk flag1"]))
df_new["Part risk flag2"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Part risk flag2"]))
df_new["Part risk flag3"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Part risk flag3"]))
df_new["Part risk flag4"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Part risk flag4"]))
df_new["Part risk flag5"] = list(map(lambda x: 0 if x=="No" else 1,df_new["Part risk flag5"]))


features = ['Current inventory level','Product Transit Time','Transit Product Amount','Forecast sales 3 months','Sales prior 1 month','amount to stock', 'Parts overdue from source', 'Source performance prior 6 month','Amount of stock orders overdue']

MMS = MinMaxScaler()
MMS.fit(df_new[features])

Normalised_df = MMS.transform(df_new[features])
Normalised_df = pd.DataFrame(Normalised_df, columns=features)

RCF = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2, random_state=10) 
#Random state to get a consistent score
RCF.fit(Normalised_df, df_new["Product Backordered"])

#Saving Model to disk pickle.dump(RCF,open('RCF.pkl','wb'))