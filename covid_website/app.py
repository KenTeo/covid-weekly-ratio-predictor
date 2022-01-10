from flask import Flask, render_template
import pandas as pd
from pandas.tseries.offsets import DateOffset
import requests
import numpy as np
import tensorflow.keras.models as tf
import pickle

app = Flask(__name__)


def weekly_cases(select, column_name):
    cases = [0, 0, 0, 0, 0, 0]
    for x in range(6, len(select)):
        weekly_avg = (select.loc[x, column_name] +
                      select.loc[x-1, column_name] +
                      select.loc[x-2, column_name] +
                      select.loc[x-3, column_name] +
                      select.loc[x-4, column_name] +
                      select.loc[x-5, column_name] +
                      select.loc[x-6, column_name])
        cases.append(weekly_avg)
    return cases


def weekly_ratio(select, column_name):
    ratio = [0.0]*13
    for x in range(13, len(select)):
        if select.loc[x-7, column_name] == 0:
            ratio.append(ratio[-1])
        else:
            avg_ratio = (select.loc[x, column_name])/select.loc[x-7, column_name]
            ratio.append(avg_ratio)
    return ratio


def data_extract():
    api_path = 'https://covidsitrep.moh.gov.sg/_dash-layout'
    moh = requests.get(api_path).json()
    date = moh['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][1]['x']
    comm_cases = moh['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][1]['y']
    dorm_cases = moh['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][3]['y']
    import_cases = moh['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][5]['y']
    d = {"date": date, "comm_cases": comm_cases,"dorm_cases": dorm_cases, "import_cases": import_cases}
    df = pd.DataFrame(data=d)
    df["comm_weekly_cases"] = weekly_cases(df, "comm_cases")
    df["comm_weekly_ratio"] = weekly_ratio(df, "comm_weekly_cases")
    df["dorm_weekly_cases"] = weekly_cases(df, "dorm_cases")
    df["dorm_weekly_ratio"] = weekly_ratio(df, "dorm_weekly_cases")
    df["import_weekly_cases"] = weekly_cases(df, "import_cases")
    df["import_weekly_ratio"] = weekly_ratio(df, "import_weekly_cases")
    return df[-14:]


def predicting(df, n_days_for_prediction):
    covid_model = tf.load_model('covid_model')
    cols = list(df)[4:10]
    print(cols)
    df_input = df[cols].astype(float)
    with open('scaler.pkl', 'rb') as handle:
        scaler = pickle.load(handle)
    df_scaled = scaler.transform(df_input)
    last_days = np.array(df_scaled)
    last_days = np.asarray(last_days).reshape(1, 14, 6)
    for x in range(n_days_for_prediction):
        days_14 = np.asarray(last_days[-1][-14:]).reshape(1, 14, 6)
        last_days = np.concatenate([last_days[0], covid_model.predict(days_14)])
        last_days = np.asarray(last_days).reshape(1, last_days.shape[0], last_days.shape[1])
    prediction = scaler.inverse_transform(last_days[-1])
    future = prediction[-n_days_for_prediction-1:]
    return future[:, 1].tolist(), future[:, 3].tolist(),future[:, 5].tolist()


@app.route('/')
def home():
    df = data_extract()
    p_comm, p_dorm, p_import = predicting(df, 7)
    p_comm2 = [0.0 if i < 0.0 else i for i in p_comm]
    p_dorm2 = [0.0 if i < 0.0 else i for i in p_dorm]
    p_import2 = [0.0 if i < 0.0 else i for i in p_import]
    labels = df["date"].tolist()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    dates = [df.index[-1] + DateOffset(days=x+1) for x in range(0, 7)]
    time = []
    for x in range(len(dates)):
        time.append(dates[x].strftime("%Y-%m-%d"))
    comm_ratio = df["comm_weekly_ratio"].tolist()
    dorm_ratio = df["dorm_weekly_ratio"].tolist()
    import_ratio = df["import_weekly_ratio"].tolist()


    return render_template('home.html', labels=labels+time, comm_ratio=comm_ratio, dorm_ratio=dorm_ratio,
                           import_ratio=import_ratio, p_comm=['NULL']*13+p_comm2, p_dorm=['NULL']*13+p_dorm2, p_import=['NULL']*13+p_import2,
                           comm=df["comm_cases"][-1], dorm=df["dorm_cases"][-1], imported=df["import_cases"][-1],
                           comm_week=df["comm_weekly_cases"][-1], dorm_week=df["dorm_weekly_cases"][-1], imported_week=df["import_weekly_cases"][-1])
