import pandas as pd
import numpy as np
import os
import warnings
import sys
warnings.filterwarnings('ignore')
import bz2file as bz2

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from math import sqrt
import joblib

def save_model_in1():
    df = pd.read_csv('master_data_after2021.csv')
    
    groups = df[['match_id','runs_scored',
    'cum_runs','cum_wickets','total_runs','total_wickets','cum_balls','balls_left','wickets_left','runs_added']].groupby('match_id')

    match_ids = df['match_id'].unique()
    last_five = []
    for id in match_ids:
        last_five.extend(groups.get_group(id).rolling(window=30).sum()['runs_scored'].values.tolist())

    df['last_five'] = last_five
    df['last_five'] = np.where(df['last_five'].isnull(),df['cum_runs'],df['last_five'])

    df2 = df
    df2 = df2[['venue','batting_team',
        'bowling_team', 'cum_runs', 'balls_left',
        'wickets_left', 'last_five','runs_added']]

    df2 = df2.reset_index(drop=True)

    X = df2.drop('runs_added',axis=1)
    y = df2['runs_added']

    venue_dict = {'Chennai':1, 'Mumbai':2, 'Ahmedabad':3, 'Delhi':4, 'AbuDhabi':5, 'Sharjah':6,
       'Dubai':7, 'BrabourneMumbai':8, 'DYPMumbai':9, 'Pune':10, 'Kolkata':11,
       'Mohali':12, 'Lucknow':13, 'Hyderabad':14, 'Bengaluru':15, 'Guwahati':16,
       'Jaipur':17, 'Dharamsala':18, 'Mullanpur':19}
    X['venue'] = X['venue'].map(venue_dict)

    team_dict = {'MI':1, 'CSK':2, 'KKR':3, 'PBKS':4, 'RCB':5, 'DC':6, 'RR':7, 'SRH':8, 'LSG':9, 'GT':10}
    X['batting_team'] = X['batting_team'].map(team_dict)
    X['bowling_team'] = X['bowling_team'].map(team_dict)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    pipe = Pipeline(steps=[
    
    ('step1',StandardScaler()),
    ('step2',XGBRegressor(n_estimators=1000,learning_rate=0.6,max_depth=8,random_state=1))
    ])
    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)

    print(r2_score(y_test,y_pred)) 
    print(mean_absolute_error(y_test,y_pred))
    print(sqrt(mean_squared_error(y_test,y_pred)))
        
    filename = "model.joblib"
    joblib.dump(pipe, filename)
    
    # def compressed_pickle(title, data):

    #     with bz2.BZ2File(title + '.pbz2', 'w') as f:
    #         pickle.dump(data, f)

    

    # compressed_pickle('xgb_model', pipe)

if __name__ == '__main__':
    save_model_in1()


    

    
