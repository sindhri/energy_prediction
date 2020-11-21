import gzip
import json_lines
import pandas as pd
from os import listdir, scandir
import numpy as np
import datetime
import random
from sklearn.linear_model import LinearRegression

def read_one_file(filename):
    data=[]
    for item in json_lines.reader(gzip.open(filename)):
        data.append(item)
    df = pd.DataFrame.from_dict(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp',inplace=True)
    return df

def get_uuid(moid):
    uuid = []
    for filename in listdir('take_home_data_challenge/' + str(moid[0])):
        uuid.append(filename.split('.')[0])
    return uuid

# read all the files
def read_all_files(uuid, moid):
    df = []
    for customer in uuid:
        for month in moid:
            filename = 'take_home_data_challenge/' + str(month) + '/' + customer + '.jsonl.gz'
            print('load', filename)
            df.append(read_one_file(filename))
    return df

def get_dfs(df):
    # merge files for the same customer together in order
    df1 = pd.concat([df[0], df[1], df[2], df[3]])
    df2 = pd.concat([df[4], df[5], df[6], df[7]])
    df3 = pd.concat([df[8], df[9], df[10], df[11]])
    return df1, df2, df3

# calculate how many seconds the compressor needs to be active to lower customer2's refrigerator by 1 degree
def get_setpoint_adjust_seconds_per_degree(df):
    temp = 42
    setpoint = 34
    by_time = df.groupby(df.index.time).mean()
    setpoint_adjust_seconds = by_time[by_time.compressor==0].index[0].second - 7.5
    setpoint_adjust_seconds_per_degree = setpoint_adjust_seconds/(temp - setpoint)
    return setpoint_adjust_seconds_per_degree

# calculate Pd2 after compensating for the setpoint change each day
def get_pd2_after_compensation(df, setpoint_adjust_seconds_per_degree):
    df['temp'] = df.setpoint.shift(1)
    df.reset_index(inplace=True)
    df.loc[df.index==0,'temp']=42
    # compensate for the compressor run for each day
    df['temp_decrease'] =  df.temp - df.setpoint
    df['compressor_total'] = df['compressor'] * 24 * 3600 # convert daily mean to seconds
    df['compressor_calibrate'] = df['temp_decrease'] * setpoint_adjust_seconds_per_degree #adjustment in seconds
    df['compressor_total_calibrated'] = df['compressor_total'] + df['compressor_calibrate']
    # calculate the Pd2 for each day
    df['door_total'] =  df['door'] * 24 * 3600 
    df['Pd2'] = df.compressor_total/df.door_total
    return df

# predict Pd2 for different setpoints
def get_reg_setpoint_Pd2(setpoint_customer, df_day):
    setpoint_customer.sort()
    setpoint_customer = setpoint_customer.reshape(-1,1)
    Pd2_at_setpoints = df_day[['Pd2', 'setpoint']].groupby(['setpoint']).mean().values
    reg_setpoint_Pd2 = LinearRegression().fit(setpoint_customer, Pd2_at_setpoints)
    Pd2 = reg_setpoint_Pd2.predict(np.array(setpoints).reshape(-1,1)).flatten()
    return Pd2, reg_setpoint_Pd2

# get gaps of customer3
def get_gaps(df):
    df_interger_index = df.copy();
    df_interger_index.reset_index(inplace=True)
    df_interger_index['gap'] = df_interger_index.timestamp.diff().astype('timedelta64[D]')
    gaps = list(df_interger_index.gap.unique())
    gaps = [x for x in gaps if str(x)!='nan' and x!=0]
    return gaps, df_interger_index

# split the df3 data by gaps/vacancy
def split_df_by_gaps(gaps, df, df_interger_index):
    df_split = []
    early_bound = 0
    for gap in gaps:
        this_index = df_interger_index[df_interger_index.gap == gap].index.values[0]
        df_split.append(df[early_bound:this_index])
        early_bound = this_index

    df_split_day = []
    for data in df_split:
        df_split_day.append(data.resample('D').mean())
    return df_split_day

# count vacant days
def count_vacant_days(df):  
    df_day_vacant = df[(df.compressor.isnull())]
    df_day_vacant['vcount'] = 1
    vacancy_count = df_day_vacant.resample('M').sum()
    return vacancy_count.vcount.to_list()

# Predict the vacant days in May
def predict_vacant_days(vacancy_count_list, month_list, month_to_predict):
    
    vacant = np.array(vacancy_count_list).reshape(-1,1)
    month = np.array(month_list).reshape(-1,1)

    reg_month_vacant = LinearRegression().fit(month, vacant)
    # The estimated vacancy for May
    vacancy_May = reg_month_vacant.predict([[month_to_predict]])[0][0]
    return vacancy_May

# predict Pd3 based on vacant days
def predict_pd_by_nonvacant(nonvacant_by_month, Pd_by_month, nonvacant_May):
 
    nonvacant_by_month = nonvacant_by_month.reshape(-1,1)
    Pd_by_month = np.array(Pd_by_month).reshape(-1,1)
    reg_nonvacant_Pd = LinearRegression().fit(nonvacant_by_month, Pd_by_month)
    Pd_May = reg_nonvacant_Pd.predict([[nonvacant_May]])[0][0]
    return Pd_May

# setpoints for the prediction problem
setpoints = [35, 40, 45]
total_days_in_May = 31

# load all the files
moid = [1,2,3,4]
uuid = get_uuid(moid)
df = read_all_files(uuid, moid)
df1, df2, df3 = get_dfs(df)

setpoint_customer1 = df1.setpoint.unique()
setpoint_customer2 = df2.setpoint.unique()
setpoint_customer3 = df3.setpoint.unique()

# Customer1
# calculate customer1 compressor/door parameter at the setpoint 43
df1_day = df1.resample('D').mean()
df1_day['Pd1'] = df1_day.compressor/df1_day.door
Pd1_43 = df1_day.Pd1.mean()

# Customer2
# calculate setpoint_adjust_seconds_per_degree from data of customer2 
setpoint_adjust_seconds_per_degree = get_setpoint_adjust_seconds_per_degree(df2)
df2_day = df2.resample('D').mean()
# calculate Pd2 under the known setpoints
get_pd2_after_compensation(df2_day, setpoint_adjust_seconds_per_degree)
# calculate Pd2 based on different setpoints
Pd2, reg_setpoint_Pd2 = get_reg_setpoint_Pd2(setpoint_customer2, df2_day)

# Customer 3
# get gaps between records
gaps, df3_interger_index = get_gaps(df3)
df3_day_split = split_df_by_gaps(gaps, df3, df3_interger_index)

# get Pd3 for the known setpoint
for data in df3_day_split:
    data['Pd3'] = data.compressor/data.door
Pd3_split = [data.Pd3.mean() for data in df3_day_split]

# count vacant days in the previous months for customer3
df3_day = df3.resample('D').mean()
vacancy_count = count_vacant_days(df3_day)
vacancy_count_list = [0] + vacancy_count
month_list = [1,2,3,4]
month_to_predict = 5
vacancy_May = predict_vacant_days(vacancy_count_list, month_list, month_to_predict)

# predict Pd3 based on vacant days
total_days = np.array([31,28,31,30])
nonvacant_by_month = total_days - np.array(vacancy_count_list)

nonvacant_May = total_days_in_May - vacancy_May
Pd3_37 = predict_pd_by_nonvacant(nonvacant_by_month, Pd3_split, nonvacant_May)

# interpolate Pd1 and Pd3 for the setpoints
Pd2_43, Pd2_37 = reg_setpoint_Pd2.predict([setpoint_customer1, setpoint_customer3])
Pd2_43 = Pd2_43[0]
Pd2_37 = Pd2_37[0]
Pd1 = Pd1_43/Pd2_43*Pd2
Pd3 = Pd3_37/Pd2_37*Pd2

# door behavior estimate
door1 = df1.door.mean()
door2 = df2.door.mean()
door3 = df3.door.mean()

# Final prediction
door = np.array([door1, door2, door3])
vacancy_in_May = np.array([0, 0, vacancy_May])
temp = np.array([df1.temp[-1], df2.temp[-1], df3.temp[-1]]) #the last temperature of April for each customer
Pd = [Pd1, Pd2, Pd3]

prediction = []
for i in [0,1,2]:
    for setpint_index, setpoint in enumerate(setpoints):
        compressor_total_hours = door[i] * Pd[i][setpint_index] * (total_days_in_May - vacancy_in_May[i]) *24
        compressor_total_seconds = compressor_total_hours * 3600 + (temp[i] - setpoint) * setpoint_adjust_seconds_per_degree 
        print('customer',str(i+1), 'setpoint',str(setpoint),'compressor active total hours in May', compressor_total_hours)
        energy = compressor_total_seconds/3600 * 0.2 # convert to kWh
        prediction.append([uuid[i],setpoint,energy])

prediction_df = pd.DataFrame(prediction, columns = ['uuid','setpoint','predictedUsage'])
print('prediction complete in prediction.csv');