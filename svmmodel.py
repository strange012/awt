from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import date, datetime, timedelta

from sklearn import svm

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pandas as pd
import pandas.io.sql as psql
from statsmodels.tsa import arima_model as smar
import sys 
from sklearn import svm, linear_model
import calendar
import json

db_config = json.load(open('db_config.json'))
db = create_engine("postgresql://{}:{}@{}:{}/{}".format(db_config['name'], db_config['user'],
                                                             db_config['host'], db_config['port'], db_config['db']))

Session = sessionmaker(db)
session = Session()

COEF = 0.2
SCALE = 1.7

def get_from_db(start, end, airport='JFK', terminal='Terminal 1', hour=12):
    intervals = [
        range(2,6),
        [6,7],
        [8,9],
        [10,11],
        [12,13],
        [14],
        [15,16],
        [17],
        [18,19],
        [20,21],
        [22,23,24,1]
    ]
    hours = arr_to_str( [x for x in intervals if hour in x][0])
    sql = "SELECT flights, total, date, all_av_time, all_max_time FROM awt where airport = '{}' and hour = any ('{}'::int[]) \
        and extract(dow from date) = {} and date <= '{}' and terminal='{}' ".format(airport, hours, start.weekday(), end.date(), terminal)
    sql = sql + "LIMIT {} OFFSET {}" 
    chunk_size = 100000
    offset = 0
    dfs = []
    while True:
        fsql = sql.format(chunk_size, offset) 
        dfs.append(psql.read_sql(fsql, db))
        offset += chunk_size
        if len(dfs[-1]) < chunk_size:
            break
    return pd.concat(dfs)

def arr_to_str(arr):
        s = '{{'
        for index, i in enumerate(arr):
            s+= str(i) + (',' if index < len(arr)-1 else '}}')
        return s

def test(start, end, airport='JFK', terminal='Terminal 1', hour=12, plot=False):
    df = get_from_db(start=start, end=end, airport=airport, terminal=terminal, hour=hour)

    max_total = df['total'].max()
    min_total = df['total'].min()
    max_flights = df['flights'].max()
    min_flights = df['flights'].min()
    max_av = df['all_av_time'].max()
    min_av = df['all_av_time'].min()
    max_max = df['all_max_time'].max()
    min_max = df['all_max_time'].min()
    X = pd.DataFrame(data={
        'total': df['total'].add(-min_total).divide(max_total-min_total),
        'flights': df['flights'].add(-min_flights).divide(max_flights-min_total),
        'date' : df['date']
        })

    X = pd.concat([X, pd.get_dummies(df['date'].map(lambda x: calendar.month_name[x.month]))], axis=1)

    y = pd.concat([df['all_av_time'].add(-min_av).divide(max_av-min_av), df['date']], axis=1)
    y_max = pd.concat([df['all_max_time'].add(-min_max).divide(max_max-min_max), df['date']], axis=1)

    X_train = X.loc[X['date'] < start].drop(columns=['date'])
    X_test = X.loc[X['date'] >= start].drop(columns=['date']).reset_index(drop=True)

    y_train =  y.loc[y['date'] < start]['all_av_time']
    y_test = y.loc[y['date'] >= start].reset_index(drop=True)['all_av_time']

    y_max_train =  y_max.loc[y_max['date'] < start]['all_max_time']
    y_max_test = y_max.loc[y_max['date'] >= start].reset_index(drop=True)['all_max_time']


    print "len test: {}; len train: {}".format(len(X_test), len(X_train))

    svr = svm.SVR(kernel="rbf",max_iter=-1)
    svr.fit(X_train,y_train)

    y_test = y_test.multiply(max_av-min_av).add(min_av)
    av_res = pd.DataFrame(data={
        'svr' : pd.Series(svr.predict(X_test)).multiply(max_av-min_av).add(min_av),
        'real' : y_test
      })
    se_svr = 0
    for index, row in av_res.iterrows():
        se_svr += ((row['svr'] - row['real'])**2)/len(av_res)
    av_value = y_test.sum()/float(len(y_test))
    se_svr = se_svr**(0.5)


    X_max_train = pd.concat([X_train, y_train], axis=1)
    X_max_test = pd.concat([X_test, av_res['svr'].add(-min_av).divide(max_av-min_av)], axis=1)
    X_max_test.rename(columns={'svr' : 'all_av_time'}, inplace=True)


    svr_max_av = svm.SVR(kernel="rbf", max_iter=-1)
    svr_max_av.fit(X_max_train, y_max_train)
    y_max_test = y_max_test.multiply(max_max-min_max).add(min_max)


    max_res = pd.DataFrame(data={
        'svr+av' : pd.Series(svr_max_av.predict(X_max_test)).multiply(max_max-min_max).add(min_max*COEF).multiply(SCALE),
        'real' : y_max_test
      })
    se_max_av = 0
    for index, row in max_res.iterrows():
        se_max_av += float(row['svr+av'] >= row['real'])/len(max_res)

    if plot:
        av_res.plot(kind='line', use_index=True)
        max_res.plot(kind='line', use_index=True)
        plt.show()

    return {
        'av' : av_value,
        'err' :  se_svr,
        'share' : se_max_av
    }

def predict(date, flights, total, airport, terminal, hour):
    ndate = datetime.strptime(date, '%Y-%m-%d')
    df = get_from_db(start=ndate, end=ndate, airport=airport, terminal=terminal, hour=hour)

    max_total = df['total'].max()
    min_total = df['total'].min()
    max_flights = df['flights'].max()
    min_flights = df['flights'].min()
    max_av = df['all_av_time'].max()
    min_av = df['all_av_time'].min()
    max_max = df['all_max_time'].max()
    min_max = df['all_max_time'].min()

    X_av = pd.DataFrame(data={
        'total': df['total'].add(-min_total).divide(max_total-min_total),
        'flights': df['flights'].add(-min_flights).divide(max_flights-min_flights)
        })
    X_av = pd.concat([X_av, pd.get_dummies(df['date'].map(lambda x: calendar.month_name[x.month]))], axis=1)

    y_av = df['all_av_time'].add(-min_av).divide(max_av-min_av)
    y_max = df['all_max_time'].add(-min_max).divide(max_max-min_av)

    pred_data = pd.DataFrame(data={
        'total' : [float(total-min_total)/(max_total-min_total)],
        'flights' : [float(flights-min_flights)/(max_flights-min_flights)]
    })

    months = pd.DataFrame(data=[[int(x==ndate.month) for x in range(0,12)]], columns=calendar.month_name[1:])
    pred_data = pd.concat([pred_data, months], axis=1)
    svr_av = svm.SVR(kernel="rbf",max_iter=-1)
    svr_av.fit(X_av,y_av)
    av_res = svr_av.predict(pred_data)

    X_max = pd.concat([X_av, y_av], axis=1)

    pred_data['all_av_time'] = av_res
    svr_max = svm.SVR(kernel="rbf", max_iter=-1)
    svr_max.fit(X_max, y_max)
    max_res = svr_max.predict(pred_data)

    return {
        'av' : av_res[0] * (max_av-min_av) + min_av,
        'max' : (max_res[0] * (max_max-min_max) + min_max*COEF)*SCALE
    } 
