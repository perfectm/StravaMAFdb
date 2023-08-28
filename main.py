#
# Importing these libraries to handle HTTP requests
# This is how we will get data from Strava API
#
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#
#Pandas, numpy will be the backbone of our data manipulation.
#Seaborn and Matplotlib are for data visualization.
#Datetime will allow Python to recognize dates as dates, not strings.
#sqlite3 will allow us to access data offline and add tags
#
import pandas as pd 
from pandas import json_normalize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import sqlalchemy
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

#
# Oura API allows us to access our sleep data from Oura Ring
#
import oura
from oura.v2 import OuraClientV2, OuraClientDataFrameV2
from oura import OuraClient, OuraClientDataFrame

#
# Streamlit will be our web framework
#
import streamlit as st


@st.cache_data

def get_strava_data():
    auth_url = "https://www.strava.com/oauth/token"
    activites_url = "https://www.strava.com/api/v3/athlete/activities"

    payload = {
        'client_id': os.environ.get("CLIENT_ID"),
        'client_secret': os.environ.get("CLIENT_SECRET"),
        'refresh_token': os.environ.get("REFRESH_TOKEN"),
        'grant_type': "refresh_token",
        'f': 'json'
    }
    
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    
    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    try:
        strava_data = requests.get(activites_url, headers=header, params=param).json()

    except:
        print("didn't get the dataset")
    return strava_data

def process_strava_data(strava_data):
    activities = json_normalize(strava_data)

    #Reduce dataframe with only columns I care about
    cols = ['name',  'type', 'distance', 'moving_time',   
            'average_speed', 'max_speed','average_heartrate','total_elevation_gain',
            'start_date_local'
        ]
    

    activities = activities[cols]      

    #Break date into start time and date
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['start_time'] = activities['start_date_local'].dt.time
    activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
    activities['month_year'] = pd.to_datetime(activities['start_date_local']).dt.to_period('M')
    activities['month_year'] = str( activities['month_year'] ) #convert to string to omit from metrics equations
    activities['moving_time'] = activities['moving_time'] /60
    activities['distance'] = activities['distance'] /1000

    activities.rename(columns = {'distance':'distance in Km'}, inplace = True)
    activities.rename(columns = {'average_speed':'Avg. MPH'}, inplace = True)


    #split runs and hikes 
    runs = activities.loc[activities['type'] == 'Run'] 
    hikes = activities.loc[activities['type'] == 'Hike'] 

    return runs, hikes

def get_oura_data():
    # begin oura import 
    readiness_url = 'https://api.ouraring.com/v2/usercollection/daily_readiness' 
    sleep_url = 'https://api.ouraring.com/v2/usercollection/sleep'
    #readiness_url = "https://api.ouraring.com/v1/readiness"
    oura_access_token =  os.environ.get("OURA_ACCESS_TOKEN")


    params={ 
        'start_date': '2021-11-01', 
        'end_date': '2023-12-01' ,
        'per_page': 200, 'page': 1
    }
        
      
    header = {'Authorization': 'Bearer ' + oura_access_token}

    try:
        readiness_json = requests.get(readiness_url, headers=header, params=params).json()
    except:
        print("didn't get the dataset")

    try:
        sleep_json = requests.get(sleep_url, headers=header, params=params).json()
    except:
        print("2nd try failed")

    oura_readiness = json_normalize(readiness_json)
    oura_sleep = json_normalize(sleep_json)

    #readiness_json
    sleep_json
    #client = OuraClientDataFrame(personal_access_token=oura_access_token)
    #oura_readiness = client.readiness_df(start='2021-11-01')
    #oura_sleep = client.sleep_df(start='2021-11-01')
    #oura_combined = client.combined_df_edited(start='2021-11-01')    
    #oura_readiness.reset_index()
    #oura_readiness['summary_date'] = oura_readiness.index
    #oura_returnable = pd.DataFrame(oura_readiness)
    #client.sleep_summary(start='2023-01-06', end='2023-12-10')
    #oura_readiness.tail()
    return oura_readiness

# begin main section
strava_data = get_strava_data()

# hiking data is much different than runs, so I break that out separately so
# that I don't compare apples to oranges
runs, hikes = process_strava_data(strava_data)

# Commented out for now -- This is where we would add sidebar navigation
add_sidebar = st.sidebar.selectbox('Make your view selection', ('Dashboard','Individual Runs', 'Oura','Health'))

#
# Create copies of the data from Strava to compare against
#
runs_diff = runs.copy()
maf_diff = runs.copy()

#
# Right now there is no tagging method to identify MAF runs so I have resorted
# to including the string 'MAF' in the workout title.  This isn't ideal but
# it works for now
#
runs_diff = runs_diff[runs_diff["name"].str.contains('MAF') == False]
maf_diff = maf_diff[maf_diff["name"].str.contains('MAF') == True]

numeric_cols = np.array((runs_diff.dtypes == 'float64') | (runs_diff.dtypes == 'int64') )


# Calculating some medians of the past 12 months on all numeric fields 
#
metric_date_12mo = runs_diff['start_date_local'].max() - pd.DateOffset(months =12)
median_agg = runs_diff[runs_diff['start_date_local'] >= metric_date_12mo].median(numeric_only=True)
#median_agg = runs_diff.median(numeric_only=True)
#runs_diff.iloc[:,numeric_cols] = (runs_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

maf_metric_date_12mo = maf_diff['start_date_local'].max() - pd.DateOffset(months =12)
maf_median_agg = maf_diff[maf_diff['start_date_local'] >= maf_metric_date_12mo].median(numeric_only=True)
maf_diff.iloc[:,numeric_cols] = (maf_diff.iloc[:,numeric_cols] - maf_median_agg).div(maf_median_agg)


#Show individual metrics 
if add_sidebar == 'Dashboard':
    
    st.write("Runs (Non-MAF) 1 Month Averages (compared against 12 month average)")
    
    runs_metrics = runs[ [ 'name','distance in Km', 'moving_time',   
            'Avg. MPH', 'max_speed','average_heartrate','total_elevation_gain',
            'start_date_local'        ] ]
    runs_metrics = runs_metrics[runs_metrics["name"].str.contains('MAF') == False]
    metric_date_6mo = runs_metrics['start_date_local'].max() - pd.DateOffset(months =1)
    metric_date_12mo = runs_metrics['start_date_local'].max() - pd.DateOffset(months =12)
    metric_medians6mo = runs_metrics[runs_metrics['start_date_local'] >= metric_date_6mo].median(numeric_only=True)
    metric_medians12mo = runs_metrics[runs_metrics['start_date_local'] >= metric_date_12mo].median(numeric_only=True)
    
    col1, col2, col3  = st.columns(3)
    columns = [col1, col2, col3 ]
    
    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label= i, value = round(metric_medians6mo[i],1), delta = "{:.2%}".format(delta))
            count += 1
            if count >= 3:
                count = 0

    st.write("MAF runs 1 Month Averages (compared against 12 month average)")
    
    MAF_metrics = runs[ [ 'name', 'distance in Km', 'moving_time',   
            'Avg. MPH', 'max_speed','average_heartrate','total_elevation_gain',
            'start_date_local'
        ] ]
    MAF_metrics = MAF_metrics[MAF_metrics["name"].str.contains('MAF') == True]

    mafmetric_date_6mo = MAF_metrics['start_date_local'].max() - pd.DateOffset(months =1)
    mafmetric_date_12mo = MAF_metrics['start_date_local'].max() - pd.DateOffset(months =12)
    mafmetric_medians6mo = MAF_metrics[MAF_metrics['start_date_local'] >= mafmetric_date_6mo].median(numeric_only=True)
    mafmetric_medians12mo = MAF_metrics[MAF_metrics['start_date_local'] >= mafmetric_date_12mo].median(numeric_only=True)
 
    
    col1, col2, col3  = st.columns(3)
    columns = [col1, col2, col3 ]
  
    count = 0
    for i in mafmetric_medians6mo.index:
        with columns[count]:
          delta = (mafmetric_medians6mo[i] - mafmetric_medians12mo[i])/mafmetric_medians12mo[i]
          st.metric(label= i, value = round(mafmetric_medians6mo[i],1), delta = "{:.2%}".format(delta))
          count += 1
          if count >= 3:
              count = 0           


#Define Functions 
def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass    



if add_sidebar == 'Individual Runs':

    short_cols = [ 'name', 'distance in Km', 'moving_time',   
    'Avg. MPH', 'max_speed','average_heartrate','total_elevation_gain'   
]
    runs_metrics_final = runs_diff.loc[:,short_cols]
    run_metrics_numeric = runs_metrics_final.median().index.tolist()
    df_to_pct = {}
    for i in run_metrics_numeric:
        df_to_pct[i] = '{:.1%}'.format

    maf_metrics_final = maf_diff.loc[:,short_cols]
    maf_metrics_numeric = maf_metrics_final.median().index.tolist()
    st.header('Runs compared to 6 month averages')
    st.dataframe(runs_metrics_final.style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct)) 
    st.header('MAF runs compared to 6 month averages')
    st.dataframe(maf_metrics_final.style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct)) 


if add_sidebar == 'Oura':    
    st.header('Oura ring data')
    oura_data = get_oura_data()

    oura_numeric_cols = np.array((oura_data.dtypes == 'float64') | (oura_data.dtypes == 'int64') )
    
    #st.write(oura_numeric_cols)

    oura_date_12mo = oura_data.tail(365)
    st.dataframe(oura_data)


    
    oura_median12_agg = oura_data.tail(365).median()
    oura_median6_agg = oura_data.tail(180).median()
    #oura_date_12mo.iloc[:,numeric_cols] = (oura_date_12mo.iloc[:,numeric_cols] - oura_median_agg).div(oura_median_agg)

    
    st.dataframe(oura_data)
    #st.dataframe(oura_median6_agg)
    
def get_health_data():
    health = pd.read_csv("~/StravaMAFdb/health.csv")
    # st.write(health.columns)
    #health.iloc[:,13] = health.iloc[:,13].apply(lambda x: x if pd.isnull(x) else float(str(x)))
 
    health_numeric_cols = np.array((health.dtypes == 'float64') | (health.dtypes == 'int64') )

    # cleaning up the empty values
    weight = health.iloc[:,[0,13]]
    weight = weight.dropna(axis=0)
    weight = weight.replace(' ', np.nan)

    sleep = health.iloc[:,[0,5,8]]
    sleep = sleep.dropna(axis=0)
    sleep = sleep.replace(' ', np.nan)
    
    mean_sleep = sleep[sleep.columns[1]].astype(float).mean()
    some_deep =  sleep[sleep.columns[2]] > 0
    deep = sleep[some_deep]
    mean_deep = deep[deep.columns[2]].astype(float).mean()
    current_sleep = sleep[sleep.columns[1]].tail(1).astype(float).item()
    current_deep = sleep[sleep.columns[2]].tail(1).astype(float).item()
    
    VO2 = health.iloc[:,[0,11]]
    VO2 = VO2.dropna(axis=0)
    VO2 = VO2.replace(' ', np.nan)    
    current_VO2 = VO2[VO2.columns[1]].tail(1).astype(float).item()
    mean_VO2 = VO2[VO2.columns[1]].astype(float).mean()

    rhr = health.iloc[:,[0,4]]
    rhr = rhr.dropna(axis=0)
    rhr = rhr.replace(' ', np.nan) 
    current_rhr = rhr[rhr.columns[1]].tail(1).astype(float).item()
    mean_rhr = rhr[rhr.columns[1]].astype(float).mean()

    max_weight = weight[weight.columns[1]].astype(float).max()
    mean_weight = weight[weight.columns[1]].astype(float).mean()
    min_weight = weight[weight.columns[1]].astype(float).min()
    current_weight = weight[weight.columns[1]].tail(1).astype(float).item()
    max_weight_date = weight['Date'][ weight[weight.columns[1]].astype(float) ==  max_weight].astype(str).item()
    min_weight_date = weight['Date'][ weight[weight.columns[1]].astype(float) ==  min_weight ].astype(str).item()
    if current_weight < max_weight:
        weight_delta = -1+(current_weight / max_weight)
    else:
        weight_delta = 1-(current_weight / max_weight)
  
    if current_sleep > mean_sleep:
        sleep_delta = -1+(current_sleep / mean_sleep)
    else:
        sleep_delta = 1-(current_sleep / mean_sleep)
        
    if current_deep > mean_deep:
        deep_delta = -1+(current_deep / mean_deep)
    else:
        deep_delta = 1-(current_deep / mean_deep)       
        
    if current_VO2 > mean_VO2:
        VO2_delta = -1+(current_VO2 / mean_VO2)
    else:
        VO2_delta = 1-(current_VO2 / mean_VO2)  

    if current_rhr > mean_rhr:
        rhr_delta = -1+(current_rhr / mean_rhr)
    else:
        rhr_delta = 1-(current_rhr / mean_rhr)  


    col1, col2, col3  = st.columns(3)
    columns = [col1, col2, col3 ]
  
              
    with columns[0]:        
        st.metric(label = 'Current Weight', value = current_weight, delta = "{:.2%}".format(weight_delta))
    with columns[1]:        
        st.metric(label = 'Last night Sleep time (hours)', value = "{:.2}".format(current_sleep), delta = "{:.2%}".format(sleep_delta))
    with columns[2]:        
        st.metric(label = 'Last night Deep Sleep (minutes)', value = current_deep*60, delta = "{:.2%}".format(deep_delta))
    with columns[0]:        
        st.metric(label = 'VO2 Max', value = current_VO2, delta = "{:.2%}".format(VO2_delta))
    with columns[1]:      
        #st.write()
        st.metric(label = 'Resting Heart Rate', value = current_rhr, delta = "{:.2%}".format(rhr_delta))
    with columns[2]:        
        st.write()
        # st.metric(label = 'TBD', value = current_deep*60, delta = "{:.2%}".format(deep_delta))

    
if add_sidebar == 'Health':
    st.header('Health data from Apple Health/Watch')    
    
    get_health_data()