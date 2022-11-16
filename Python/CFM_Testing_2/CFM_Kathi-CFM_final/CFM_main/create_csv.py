import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import timedelta

start = '01-01-1000 00:00:00'
end = '31-12-1001 00:00:00'
time_step = 1           # step_length [min]
abs_zero_T = 273.15     # absolute zero [K]
filename = 'CFMinput/NGRIP'


data_NGRIP = {'mean_temperature_C:': -31.1, 'mean_temperature_K': abs_zero_T-31.1, 'accumulation_rate': 0.19}
data_DomeC = {'mean_temperature_C:': -54, 'mean_temperature_K': abs_zero_T-54, 'accumulation_rate': 0.0295}


def set_time(start, end):
    times = np.arange(start, end, 1)
    return times, np.shape(times)[0]

    #span = pd.period_range(start, end, freq='H')
    #span = pd.to_timestamp(span, error='coerce')
    #return span, len(span)

    #d = datetime.strptime(start, '%d-%m-%Y %H:%M:%S')
    #list_dates = []
    #while d != datetime.strptime(end, '%d-%m-%Y %H:%M:%S'):
    #    list_dates.append(d)
    #    d += timedelta(hours=time_step)
    #    print('d:', d)
    #return list_dates, len(list_dates)

def set_steady_state_T(temp, no_steps):
    return np.ones([no_steps]) * temp

def set_steady_state_acc(acc, no_steps):
    return np.ones([no_steps]) * acc

def create_csv(time, temperature, accumulation, filename):
    data = pd.DataFrame({'time': time, 'TSKIN': temperature, 'BDOT': accumulation}).set_index('time')
    #data.index = pd.to_datetime(data.index)
    #print('data:', data)
    return data.to_csv(filename + '.csv')

def create_pkl(time, temperature, accumulation, filename):
    data = pd.DataFrame({'time': time, 'TSKIN': temperature, 'BDOT': accumulation}).set_index('time')
    #data.index = pd.to_datetime(data.index)
    #print(type(data.index))
    return data.to_pickle(filename + '.pkl')


times, no_steps = set_time(1000.00, 2020.00)
#print(pd.to_datetime(times[0]))
#print(times)
#times2, no_steps2 = set_time(start2, end2)
#times = times1 + times2
#times = np.array(times)

#no_steps = no_steps1 + no_steps2

temperature = set_steady_state_T(data_NGRIP.get('mean_temperature_K'), no_steps)
accumulation = set_steady_state_acc(data_NGRIP.get('accumulation_rate'), no_steps)

create_csv(times, temperature, accumulation, filename)   # to check how the .pkl looks like
create_pkl(times, temperature, accumulation, filename)








