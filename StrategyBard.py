import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import talib
import os
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
import mplfinance as mpf

def calculate_trend_following_indicators(combination, df):
    if 'SMA' in combination:
        if df['SMA'][-1] > df['Close'][-1]:
            signal = 1
        elif df['SMA'][-1] < df['Close'][-1]:
            signal = -1
        else: 
            signal = 0
    elif 'EMA' in combination:
        if df['EMA'][-1] > df['Close'][-1]:
            signal = 1
        elif df['EMA'][-1] < df['Close'][-1]:
            signal = -1
        else: 
            signal = 0
    elif 'WMA' in combination:
        if df['WMA'][-1] > df['Close'][-1]:
            signal = 1
        elif df['WMA'][-1] < df['Close'][-1]:
            signal = -1
        else: 
            signal = 0
    elif 'MACD' in combination:
        if df['MACD'][-1] > df['SignalMACD'][-1] and df['HistogramMACD'][-1] > 0:
            signal = 1
        elif df['MACD'][-1] < df['SignalMACD'][-1] and df['HistogramMACD'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'ADX' in combination:
        if df['ADX'][-1] > 25:
            signal = 1
        else:
            signal = 0
    elif 'DMI' in combination:
        if df['DMI'][-1] > 50:
            signal = 1
        else:
            signal = 0        
    return signal

def calculate_oscillators(combination, df):
    if 'RSI' in combination:
        if df['RSI'][-1] > 70:
            signal = -1
        elif df['RSI'][-1] < 30:
            signal = 1
        else:
            signal = 0
    elif 'STOCH' in combination:
        if df['STOCH_K'][-1] > df['STOCH_D'][-1]:
            signal = 1
        elif df['STOCH_K'][-1] < df['STOCH_D'][-1]:
            signal = -1
        else:
            signal = 0
    elif 'WILLR' in combination:
        if df['WILLR'][-1] < -20:
            signal = 1
        elif df['WILLR'][-1] > -80:
            signal = -1
        else:
            signal = 0
    elif 'UO' in combination:
        if df['UO'][-1] > 70:
            signal = -1
        elif df['UO'][-1] < 30:
            signal = 1
        else:
            signal = 0
    elif 'COPK' in combination:
        if df['COPK'][-1] > 0:
            signal = 1
        elif df['COPK'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'DPO' in combination:
        if df['DPO'][-1] > 0:
            signal = 1
        elif df['DPO'][-1] < 0:
            signal = -1
        else:
            signal = 0
    else:
        signal = 0
        
    return signal

def calculate_volume_indicators(combination, df):
    if 'OBV' in combination:
        if df['OBV'][-1] > df['OBV'][-2]:
            signal = 1
        elif df['OBV'][-1] < df['OBV'][-2]:
            signal = -1
        else:
            signal = 0
    elif 'ADL' in combination:
        if df['ADL'][-1] > df['ADL'][-2]:
            signal = 1
        elif df['ADL'][-1] < df['ADL'][-2]:
            signal = -1
        else:
            signal = 0
    elif 'CMF' in combination:
        if df['CMF'][-1] > 0:
            signal = 1
        elif df['CMF'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'FI' in combination:
        if df['FI'][-1] > 0:
            signal = 1
        elif df['FI'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'EOM' in combination:
        if df['EOM'][-1] > 0:
            signal = 1
        elif df['EOM'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'VO' in combination:
        if df['VO'][-1] > df['VO'][-2]:
            signal = 1
        elif df['VO'][-1] < df['VO'][-2]:
            signal = -1
        else:
            signal = 0
    else:
        signal = 0
        
    return signal

def calculate_volatility_indicators(combination, df):
    if 'BBANDS' in combination:
        if df['Close'][-1] > df['BBANDS_U'][-1]:
            signal = -1
        elif df['Close'][-1] < df['BBANDS_L'][-1]:
            signal = 1
        else:
            signal = 0
    elif 'KBANDS' in combination:
        if df['Close'][-1] > df['KBANDS_U'][-1]:
            signal = -1
        elif df['Close'][-1] < df['KBANDS_L'][-1]:
            signal = 1
        else:
            signal = 0
    elif 'DOCH' in combination:
        if df['DOCH'][-1] > 0:
            signal = 1
        elif df['DOCH'][-1] < 0:
            signal = -1
        else:
            signal = 0
    elif 'ATR' in combination:
        if df['ATR'][-1] > df['ATR'][-2]:
            signal = 1
        elif df['ATR'][-1] < df['ATR'][-2]:
            signal = -1
        else:
            signal = 0
    elif 'RVI' in combination:
        if df['RVI'][-1] > df['RVI'][-2]:
            signal = 1
        elif df['RVI'][-1] < df['RVI'][-2]:
            signal = -1
        else:
            signal = 0
    elif 'STD' in combination:
        if df['STD'][-1] > df['STD'][-2]:
            signal = 1
        elif df['STD'][-1] < df['STD'][-2]:
            signal = -1
        else:
            signal = 0
    else:
        signal = 0
        
    return signal

# Trend Following Indicators
trend_following_indicators = ['SMA','EMA','WMA','MACD','ADX','DMI']
# Oscillators
oscillators = ['RSI','STOCH','WILLR','UO','COPK','DPO']
# Volume Indicators
volume_indicators = ['OBV','ADL','CMF','FI','EOM','VO']
# Volatility Indicators
volatility_indicators = ['BBANDS','KBANDS','DOCH','ATR','RVI','STD']

# Import data
symbol = 'GC=F'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
file_name = 'data.csv'

if os.path.exists(file_name):
    file_time = os.path.getmtime(file_name)
    if (time.time() - file_time) / 60 > 15:
        dl_data = yf.download(symbol, start=start_date, end=end_date)
        dl_data.to_csv(file_name)
else:
    dl_data = yf.download(symbol, start=start_date, end=end_date)
    dl_data.to_csv(file_name)

df = pd.read_csv(file_name, index_col='Date')
# Calculate trend following indicators
df['SMA'] = talib.SMA(df['Close'], timeperiod=10)
df['EMA'] = talib.EMA(df['Close'], timeperiod=10)
df['WMA'] = talib.WMA(df['Close'], timeperiod=10)
df['MACD'],df['SignalMACD'],df['HistogramMACD'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
# Calculate the Upward Directional Movement (DM+) and Downward Directional Movement (DM-)
df['DM+'] = df['High'] - df['High'].shift(1)
df['DM-'] = df['Low'].shift(1) - df['Low']
# Calculate the Positive Directional Indicator (DI+) and Negative Directional Indicator (DI-)
df['DI+'] = df['DM+'].ewm(span=14, min_periods=14).mean()
df['DI-'] = df['DM-'].ewm(span=14, min_periods=14).mean()
# Calculate the Directional Movement Index (DMI)
df['DMI'] = 100 * (df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])
# Calculate oscillators
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['STOCH_K'],df['STOCH_D']  = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowd_period=3)
df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
# Calculate the Ultimate Oscillator (UO)
df['UO'] = 100 * (talib.SMA(talib.TYPPRICE(df['High'], df['Low'], df['Close']), timeperiod=14) - talib.SMA(talib.TYPPRICE(df['High'], df['Low'], df['Close']), timeperiod=14).shift(14)) / talib.SMA(talib.TYPPRICE(df['High'], df['Low'], df['Close']), timeperiod=14)
df['COPK'] = talib.CMO(df['Close'], timeperiod=14)
df['DPO'] = talib.SMA(df['Close'], timeperiod=20) - talib.SMA(df['Close'], timeperiod=20).shift(20)
# Calculate volume indicators
df['OBV'] = talib.OBV(df['Close'], df['Volume'])
df['ADL'] = talib.TYPPRICE(df['High'], df['Low'], df['Close']) * df['Volume']
# Calculate the Money Flow Volume (MFV)
df['MFV'] = df['Close'] * df['Volume']
# Calculate the Chaikin Money Flow (CMF)
df['CMF'] = (df['MFV'] - df['MFV'].shift(14)) / df['MFV'].mean()
# Calculate the Force Index (FI)
df['FI'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
# Calculate the midpoint for each period
df['Midpoint'] = (df['High'] + df['Low']) / 2
# Calculate the midpoint move for each period
df['MidpointMove'] = df['Midpoint'] - df['Midpoint'].shift(1)
# Calculate the box ratio for each period
df['BoxRatio'] = df['Volume'] / (df['High'] - df['Low'])
# Calculate the EOM indicator for each period
df['EOM'] = df['MidpointMove'] / df['BoxRatio']
df['VO'] = (df['Volume'].rolling(14).mean() - df['Volume'].rolling(28).mean()) / df['Volume'].rolling(28).mean()
# Calculate volatility indicators
df['BBANDS_U'],df['BBANDS_M'],df['BBANDS_L'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
df['KBANDS_MA'] = df['Close'].rolling(window=20).mean()
df['KBANDS_U'] = df['Close'].rolling(window=20).mean() + 2 * (df['Close'].rolling(window=50).mean() - df['Close'].rolling(window=20).mean())
df['KBANDS_L'] = df['Close'].rolling(window=20).mean() - 2 * (df['Close'].rolling(window=50).mean() - df['Close'].rolling(window=20).mean())
df['DOCH'] = df['Close'].rolling(window=14).mean() - df['Close'].rolling(window=42).mean()
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['RVI'] = df['Close'].diff(14) / (df['High'].diff(14) - df['Low'].diff(14)).fillna(0)
df['STD'] = np.std(df['Close'].diff(14))
# Get all possible combinations of indicators
combinations = list(itertools.product(trend_following_indicators, oscillators, volume_indicators, volatility_indicators))

signals = []

# Iterate through combinations
for combination in combinations:
    combination_signals = [
        calculate_trend_following_indicators(combination, df),
        calculate_oscillators(combination, df),
        calculate_volume_indicators(combination, df),
        calculate_volatility_indicators(combination, df)]
    signals.append(( combination_signals))

signal_data = signals
# Create a figure and axes for subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Line Plot
axes[0, 0].plot(signal_data)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Signal Value')
axes[0, 0].set_title('Line Plot of Signal')

# Scatter Plot
x = np.arange(len(signal_data))  # Create a sequence of x values
y = [sum(sublist) for sublist in signal_data]
axes[0, 1].scatter(x, y)
axes[0, 1].set_xlabel('Data Index')
axes[0, 1].set_ylabel('Signal Value')
axes[0, 1].set_title('Scatter Plot of Signal')

# Bar Plot
unique_values, counts = np.unique(signal_data, return_counts=True)
axes[0, 2].bar(unique_values, counts)
axes[0, 2].set_xlabel('Signal Value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Bar Plot of Signal')

# Histogram
axes[1, 0].hist(signal_data, bins='auto')
axes[1, 0].set_xlabel('Signal Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram of Signal')

# Spectrogram
# Assuming signal_data is a 2D array representing the spectrogram
axes[1, 1].imshow(signal_data, aspect='auto', cmap='coolwarm')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Spectrogram of Signal')
plt.colorbar(axes[1, 1].imshow(signal_data, aspect='auto', cmap='coolwarm'), ax=axes[1, 1])

# Pie Chart
labels = ['Sell', 'Hold', 'Buy']
data_values = [sum([1 for sublist in signal_data if value in sublist]) for value in [-1, 0, 1]]
axes[1, 2].pie(data_values, labels=labels, autopct='%1.1f%%')
axes[1, 2].set_title('Pie Chart')

# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()
# print(signal_data)





