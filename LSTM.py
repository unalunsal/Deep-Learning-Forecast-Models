# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:38:16 2019
@author: unalunsal
Data Source:
https://www.eia.gov/totalenergy/data/browser/?tbl=T09.09#/?f=M
"""

import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(13)
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure, output_file


df = pd.read_csv(r'C:\Users\unalu\Desktop\eia\MER_T09_09.csv')
df['YYYYMM'] = df['YYYYMM'].astype(str).values # convert integer to string 

df = df.loc[~df['YYYYMM'].str.endswith('13'),:]  # drop the annual values 
 
df['YYYYMM'] = pd.to_datetime(df['YYYYMM'], format='%Y%m', errors='coerce').dropna() # convert string to datetime

df_clerdus = df.loc[df['MSN'] == 'CLERDUS',]

dates = df_clerdus['YYYYMM']

data_ts = df_clerdus['Value'].values.astype(np.float)

# Model parameters 
history_size = 20
train_size = len(data_ts) - 1 - history_size
train_begin = 530    #time period to start out of sample forecasting
step_ahead = 0
batch_size = 256
buffer_size = 10000
eval_interval = 200
epochs = 20 #10


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

lstm_pred = np.array([])    #np array to save LSTM forecast values 

for i in range(250, train_size):  # loop for recursive out of sample forecast values
    mean = data_ts[:i].mean()     # mean of the training set
    std = data_ts[:i].std()       # std of the training set 
    data_ts_train = (data_ts - mean) / std  # standardize the training set
   
    x_train_uni, y_train_uni = univariate_data(data_ts_train, 0, train_size, history_size, step_ahead)
    x_val_uni, y_val_uni = univariate_data(data_ts_train, train_size, None, history_size, step_ahead)

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(batch_size).repeat()

    mod_lstm = tf.keras.models.Sequential([ tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
                                            tf.keras.layers.Dense(1) ])
    
    mod_lstm.compile(optimizer='adam', loss='mae')
    
    mod_lstm.fit(train_univariate, epochs=epochs, steps_per_epoch = eval_interval,
                          validation_data=val_univariate, validation_steps=50)
    for x, y in val_univariate.take(3):
        forecast = (mod_lstm.predict(x)[0].item() * std) + mean
    lstm_pred = np.append(lstm_pred, forecast)


lstm_pred = np.around(lstm_pred, decimals = 2)

forecast_ts = df_clerdus['Value'][:(data_ts.shape[0] - (train_size - train_begin))].append(pd.Series(["".join(item) for item in lstm_pred.astype(str)]))

forecast_ts.index = np.arange(0, len(forecast_ts),1)

source = ColumnDataSource(data=dict(date=dates, close=df_clerdus['Value']))
source_lstm = ColumnDataSource(data=dict(date=dates, close=forecast_ts))

TOOLS="hover,pan,box_zoom,undo,reset,save"
width = 800
p = figure(plot_height=500, plot_width=width, title="Cost of Fossil-Fuel Receipts at Electric Generating Plants, Monthly", 
           tools=TOOLS, x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#FFFFFF", x_range=(dates[150], dates[500]))

p.toolbar.logo = None
p.line('date', 'close', source=source_lstm, color = 'blue',  name="LSTM Forecast", legend = 'LSTM Forecast Values')
p.line('date', 'close', source=source, color = 'green', name = "Actual Value", legend = 'Actual Values')

p.legend.location = "top_left"
p.legend.click_policy="hide"

p.yaxis.axis_label = 'Dollars per Million Btu, Including Taxes'

p.hover.tooltips = [ ("Value", "$y Dollars per Million Btu, Including Taxes")]   
p.hover.formatters = {'Value' : 'printf'}
#p.hover.mode='vline'  


select = figure(title="Change the range above",
                plot_height=100, plot_width=width, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color="#FFFFFF")

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "gray"
range_tool.overlay.fill_alpha = 0.4

select.line('date', 'close', source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool

output_file("LSTM_ForecastModel.html")


show(column(p, select))
