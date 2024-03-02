import yfinance as yf
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_ta as pta
from pandas_ta import *
import ta.trend as tr
import ta.volatility as vo
import ta.momentum as mo
from ta.volume import MFIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.momentum import ROCIndicator
from ta.volume import money_flow_index
from ta.volume import AccDistIndexIndicator
from ta.volume import EaseOfMovementIndicator
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import SMAIndicator
from ta.trend import MACD
from ta.momentum import StochasticOscillator


import talib
from stockstats import StockDataFrame as Sdf

# Define the strategies functions for each indicator combination
def perceptron_strategy(data):
    return 0#np.random.choice([1, -1, 0])

def sma_sma_strategy(data):
    # Load data from CSV and calculate SMA values
    df = data
    sma_50 = df['Adj Close'].rolling(window=50).mean()
    sma_200 = df['Adj Close'].rolling(window=200).mean()

    # Apply Golden Cross strategy
    buy_signal = (sma_50.iloc[-1] > sma_200.iloc[-1]) and (sma_50.iloc[-2] <= sma_200.iloc[-2])
    sell_signal = (sma_50.iloc[-1] < sma_200.iloc[-1]) and (sma_50.iloc[-2] >= sma_200.iloc[-2])
    hold_signal = (sma_50.iloc[-1] == sma_200.iloc[-1]) 
    # print(buy_signal,sell_signal,hold_signal)
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_ema_strategy(data):
    # Load data from CSV and calculate EMA values
    df = data
    ema_50 = df['Adj Close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['Adj Close'].ewm(span=200, adjust=False).mean()

    # Apply EMA-EMA crossover strategy
    buy_signal = (ema_50.iloc[-1] > ema_200.iloc[-1]) and (ema_50.iloc[-2] <= ema_200.iloc[-2])
    sell_signal = (ema_50.iloc[-1] < ema_200.iloc[-1]) and (ema_50.iloc[-2] >= ema_200.iloc[-2])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_ema_strategy(data):
    # Load data from CSV and calculate SMA and EMA values
    df = data
    sma_50 = df['Adj Close'].rolling(window=50).mean()
    ema_200 = df['Adj Close'].ewm(span=200, adjust=False).mean()

    # Apply SMA-EMA crossover strategy
    buy_signal = (sma_50.iloc[-1] > ema_200.iloc[-1]) and (sma_50.iloc[-2] <= ema_200.iloc[-2])
    sell_signal = (sma_50.iloc[-1] < ema_200.iloc[-1]) and (sma_50.iloc[-2] >= ema_200.iloc[-2])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_macd_strategy(data):
    # Load data from CSV and calculate SMA and MACD values
    df = data
    sma_50 = df['Adj Close'].rolling(window=50).mean()
    ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    # Apply SMA-MACD crossover strategy
    buy_signal = (sma_50.iloc[-1] > signal_line.iloc[-1]) and (sma_50.iloc[-2] <= signal_line.iloc[-2]) and (macd.iloc[-1] > signal_line.iloc[-1])
    sell_signal = (sma_50.iloc[-1] < signal_line.iloc[-1]) and (sma_50.iloc[-2] >= signal_line.iloc[-2]) and (macd.iloc[-1] < signal_line.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_psar_strategy(data):
    # Load data from CSV and calculate SMA and PSAR values
    df = data
    sma_50 = df['Adj Close'].rolling(window=50).mean()
    psar = tr.PSARIndicator(high=df['High'], low=df['Low'], close=df['Adj Close'], step=0.02, max_step=0.2)

    # Apply SMA-PSAR crossover strategy
    buy_signal = (sma_50.iloc[-1] > psar.psar_up().iloc[-1]) and (sma_50.iloc[-2] <= psar.psar_down().iloc[-2])
    sell_signal = (sma_50.iloc[-1] < psar.psar_down().iloc[-1]) and (sma_50.iloc[-2] >= psar.psar_up().iloc[-2])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0   

def sma_ichimoku_strategy(data):
    # Load data from CSV and calculate SMA and Ichimoku values
    df = data
    sma_26 = df['Adj Close'].rolling(window=26).mean()
    sma_52 = df['Adj Close'].rolling(window=52).mean()
    conversion_line = (sma_26 + sma_52) / 2
    base_line = df['Adj Close'].rolling(window=104).mean()
    leading_span_a = (conversion_line + base_line) / 2
    leading_span_b = df['Adj Close'].rolling(window=156).mean().shift(26)
    chikou_span = df['Adj Close'].shift(-26)

    # Apply SMA-Ichimoku crossover strategy
    buy_signal = (conversion_line.iloc[-1] > base_line.iloc[-1]) and (chikou_span.iloc[-1] > leading_span_a.iloc[-1]) and (chikou_span.iloc[-1] > leading_span_b.iloc[-1])
    sell_signal = (conversion_line.iloc[-1] < base_line.iloc[-1]) and (chikou_span.iloc[-1] < leading_span_a.iloc[-1]) and (chikou_span.iloc[-1] < leading_span_b.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def sma_supertrend_strategy(data):
    # Load data from CSV and calculate SMA and Supertrend values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    atr = vo.average_true_range(df['High'], df['Low'], df['Adj Close'], window=10)
    basic_upper_band = (df['High'] + df['Low']) / 2 + 3 * atr
    basic_lower_band = (df['High'] + df['Low']) / 2 - 3 * atr
    final_upper_band = pd.Series(index=basic_upper_band.index)
    final_lower_band = pd.Series(index=basic_lower_band.index)
    supertrend = pd.Series(index=df.index)

    for i in range(1, len(df)):
        if pd.isna(final_upper_band.iloc[i-1]):
            final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            final_lower_band.iloc[i] = basic_lower_band.iloc[i]
        else:
            if df['Adj Close'].iloc[i] > final_upper_band.iloc[i-1]:
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
            if df['Adj Close'].iloc[i] < final_lower_band.iloc[i-1]:
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        if i < 10:
            supertrend.iloc[i] = 0
        else:
            if supertrend.iloc[i-1] == final_upper_band.iloc[i-1]:
                if df['Adj Close'].iloc[i] <= final_upper_band.iloc[i]:
                    supertrend.iloc[i] = final_upper_band.iloc[i-1]
                else:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
            else:
                if df['Adj Close'].iloc[i] >= final_lower_band.iloc[i]:
                    supertrend.iloc[i] = final_lower_band.iloc[i-1]
                else:
                    supertrend.iloc[i] = final_upper_band.iloc[i]

    # Apply SMA-Supertrend strategy
    buy_signal = (df['Adj Close'].iloc[-1] > supertrend.iloc[-1]) and (df['Adj Close'].iloc[-1] > sma.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < supertrend.iloc[-1]) or (df['Adj Close'].iloc[-1] < sma.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_rsi_strategy(data):
    # Load data from CSV and calculate SMA and RSI values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    rsi = mo.RSIIndicator(df['Adj Close'], window=14).rsi()

    # Apply SMA-RSI strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (rsi.iloc[-1] > 50)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (rsi.iloc[-1] < 50)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0
    
def sma_stochastic_strategy(data):
    # Load data from CSV and calculate SMA and Stochastic values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    stochastic = mo.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch()

    # Apply SMA-Stochastic strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (stochastic.iloc[-1] < 20)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (stochastic.iloc[-1] > 80)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_cci_strategy(data):
    # Load data from CSV and calculate SMA and CCI values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    cci = tr.CCIIndicator(df['High'], df['Low'], df['Close'], window=20)

    # Apply SMA-CCI strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (cci.cci()[-1] > 100)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (cci.cci()[-1] < -100)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_roc_strategy(data):
    # Load data from CSV and calculate SMA and ROC values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    roc = mo.ROCIndicator(df['Adj Close'], window=10)

    # Apply SMA-ROC strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (roc.roc()[-1] > 0)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (roc.roc()[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_wpr_strategy(data):
    # Load data from CSV and calculate SMA and WPR values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    wpr = mo.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14)

    # Apply SMA-WPR strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (wpr.williams_r()[-1] > -80)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (wpr.williams_r()[-1] < -20)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_macd_hist_strategy(data):
    # Load data from CSV and calculate SMA and MACD Histogram values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    # macd = mo.MACD(df['Adj Close'])
    # ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    # ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    # macd = ema_12 - ema_26

    # macd_hist = tr.MACDHistogram(macd)
    macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Apply SMA-MACD-Hist strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (macd_hist[-1] > 0) and (macd_signal[-1] > 0)
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (macd_hist[-1] < 0) or (macd_signal[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_bbands_strategy(data):
    # Load data from CSV and calculate SMA and Bollinger Bands values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    bbands = vo.BollingerBands(df['Adj Close'], window=20)

    # Apply SMA-Bollinger Bands strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (df['Adj Close'].iloc[-1] < bbands.bollinger_hband()[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (df['Adj Close'].iloc[-1] > bbands.bollinger_lband()[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_atr_strategy(data):
    # Load data from CSV and calculate SMA and ATR values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    atr = vo.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)

    # Apply SMA-ATR strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (df['Low'].iloc[-1] > (sma.iloc[-1] - atr.average_true_range()[-1]))
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (df['High'].iloc[-1] < (sma.iloc[-1] + atr.average_true_range()[-1]))


    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_stdev_strategy(data):
    # Load data from CSV and calculate SMA and Standard Deviation values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    stdev = df['Adj Close'].rolling(window=20).std()

    # Apply SMA-StDev strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (df['Adj Close'].iloc[-1] < (sma.iloc[-1] + (2 * stdev.iloc[-1])))
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (df['Adj Close'].iloc[-1] > (sma.iloc[-1] + (2 * stdev.iloc[-1])))

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_kc_strategy(data):
    # Load data from CSV and calculate SMA and Keltner Channel values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    kc = vo.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20, window_atr=10)
    
    # Apply SMA-KC strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (df['Low'].iloc[-1] > kc.keltner_channel_lband()[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (df['High'].iloc[-1] < kc.keltner_channel_hband()[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_donchian_strategy(data):
    # Load data from CSV and calculate SMA and Donchian Channel values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    # dc = tr.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
    upperband = df['High'].rolling(window=20).max()
    lowerband = df['Low'].rolling(window=20).min()
    # df['Donchian_middle'] = (df['Donchian_upper'] + df['Donchian_lower']) / 2    
    # Apply SMA-Donchian strategy
    buy_signal = (df['Adj Close'].iloc[-1] > sma.iloc[-1]) and (df['Close'].iloc[-1] > upperband.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < sma.iloc[-1]) or (df['Close'].iloc[-1] < lowerband.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_chandelier_exit_strategy(data):
    # Load data from CSV and calculate SMA and Chandelier Exit values
    df = data
    sdf = Sdf.retype(df)
    # sdf.drop_duplicates()
    sma = sdf['adj close'].rolling(window=20).mean()
    chandelier_exit = sdf['close_22_ema'] - 3 * sdf['atr']

    # Apply SMA-Chandelier Exit strategy
    buy_signal = sdf['adj close'].iloc[-1] > sma.iloc[-1]
    sell_signal = sdf['adj close'].iloc[-1] < chandelier_exit.iloc[-1]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0
    

def sma_obv_strategy(data):
    # Load data from CSV and calculate SMA and OBV values
    df = data
    df.columns = [x.title() for x in df.columns]

    sma = df['Adj Close'].rolling(window=20).mean()
    obv = [0]
    for i in range(1, len(df)):
        if df['Adj Close'][i] > df['Adj Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Adj Close'][i] < df['Adj Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Apply SMA-OBV strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['OBV'].iloc[-1] > df['OBV'].iloc[-2]
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['OBV'].iloc[-1] < df['OBV'].iloc[-2]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_cmf_strategy(data):
    # Load data from CSV and calculate SMA and CMF values
    df = data
    df.columns = [x.title() for x in df.columns]
    # print(df)    
    sma = df['Adj Close'].rolling(window=20).mean()
    cmf = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow()
    df['CMF'] = cmf

    # Apply SMA-CMF strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['CMF'].iloc[-1] > 0
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['CMF'].iloc[-1] < 0

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0        

def sma_vroc_strategy(data):
    # Load data from CSV and calculate SMA and VROC values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    roc = ROCIndicator(close=df['Volume'], window=20).roc()
    df['VROC'] = roc

    # Apply SMA-VROC strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['VROC'].iloc[-1] > 0
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['VROC'].iloc[-1] < 0

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_mfi_strategy(data):
    # Load data from CSV and calculate SMA and MFI values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    mfi = money_flow_index(
        high=df['High'], low=df['Low'], close=df['Adj Close'], volume=df['Volume'], window=20
    )
    df['MFI'] = mfi

    # Apply SMA-MFI strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['MFI'].iloc[-1] < 20
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['MFI'].iloc[-1] > 80

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_adl_strategy(data):
    # Load data from CSV and calculate SMA and ADL values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    adl = AccDistIndexIndicator(
        high=df['High'], low=df['Low'], close=df['Adj Close'], volume=df['Volume'], fillna=False
    ).acc_dist_index()
    df['ADL'] = adl

    # Apply SMA-ADL strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['ADL'].iloc[-1] < df['ADL'].rolling(window=20).mean().iloc[-1]
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['ADL'].iloc[-1] > df['ADL'].rolling(window=20).mean().iloc[-1]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def sma_eom_strategy(data):
    # Load data from CSV and calculate SMA and EOM values
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()
    eom = EaseOfMovementIndicator(
        high=df['High'], low=df['Low'], volume=df['Volume'], window=14, fillna=False
    ).sma_ease_of_movement()

    # Apply SMA-EOM strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and eom.iloc[-1] < eom.rolling(window=20).mean().iloc[-1]
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and eom.iloc[-1] > eom.rolling(window=20).mean().iloc[-1]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def sma_pivot_points_strategy(data):
    # Load data from CSV and calculate SMA
    df = data
    sma = df['Adj Close'].rolling(window=20).mean()

    # Calculate Pivot Points
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    close = df['Adj Close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high

    # Apply SMA-Pivot Points strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['Low'].iloc[-1] > r1
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['High'].iloc[-1] < s1

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

    # Load data from CSV and calculate SMA
    df = pd.read_csv(data)
    sma = df['Adj Close'].rolling(window=20).mean()

    # Calculate Fibonacci retracement levels
    high = df['High'].iloc[-1]
    low = df['Low'].iloc[-1]
    fib_levels = talib.fibonacci_retracement(high, low)

    # Apply SMA-Fibonacci Retracement strategy
    buy_signal = df['Adj Close'].iloc[-1] > sma.iloc[-1] and df['Adj Close'].iloc[-1] > fib_levels[0]
    sell_signal = df['Adj Close'].iloc[-1] < sma.iloc[-1] and df['Adj Close'].iloc[-1] < fib_levels[2]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def sma_fibonacci_retracement_strategy(data, sma_period=20, fib_levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    # Calculate SMA
    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
    
    # Calculate max and min price
    max_price = data['High'].max()
    min_price = data['Low'].min()
    
    # Calculate Fibonacci retracement levels
    diff = max_price - min_price
    fib_levels_price = []
    for level in fib_levels:
        fib_levels_price.append(max_price - diff * level)
        
    # Check if price crosses above or below a Fibonacci retracement level
    signals = []
    for i in range(len(data)):
        if data['Close'][i] > data['SMA'][i]:
            for level in fib_levels_price:
                if data['Close'][i] > level and data['Close'][i-1] <= level:
                    signals.append(1)  # Buy signal
                    break
            else:
                signals.append(0)  # No signal
        else:
            for level in fib_levels_price:
                if data['Close'][i] < level and data['Close'][i-1] >= level:
                    signals.append(-1)  # Sell signal
                    break
            else:
                signals.append(0)  # No signal
                
    return signals[-1]

def sma_srl_strategy(data, sma_periods=[20, 50]):
    # Compute SMAs
    for period in sma_periods:
        sma_label = f"SMA_{period}"
        data[sma_label] = data['Close'].rolling(period).mean()

    # Compute potential support and resistance levels
    data['SRL'] = 0
    for i in range(max(sma_periods), len(data)):
        if data['Close'][i] < data['SMA_%d' % sma_periods[0]][i] and data['Close'][i] > data['SMA_%d' % sma_periods[1]][i]:
            data['SRL'][i] = 1  # Resistance level
        elif data['Close'][i] > data['SMA_%d' % sma_periods[0]][i] and data['Close'][i] < data['SMA_%d' % sma_periods[1]][i]:
            data['SRL'][i] = -1  # Support level
	
	# data['SRL'].fillna(0, inplace=True)
    # Return last signal
    return int(data['SRL'][-1])

def sma_gann_lines_strategy(data):
    # Calculate SMA
    data['SMA'] = data['Adj Close'].rolling(window=20).mean()
    
    # Calculate the highest high and lowest low over the past 20 days
    data['HH'] = data['High'].rolling(window=20).max()
    data['LL'] = data['Low'].rolling(window=20).min()
    
    # Calculate the pivot point and the Gann lines
    data['PP'] = (data['HH'] + data['LL'] + data['SMA']) / 3
    data['R1'] = 2 * data['PP'] - data['LL']
    data['S1'] = 2 * data['PP'] - data['HH']
    data['R2'] = data['PP'] + data['HH'] - data['LL']
    data['S2'] = data['PP'] - data['HH'] + data['LL']
    data['R3'] = data['HH'] + 2 * (data['PP'] - data['LL'])
    data['S3'] = data['LL'] - 2 * (data['HH'] - data['PP'])
    
    # Calculate the signal
    data['Signal'] = 0
    for i in range(1, len(data)):
        if data['Adj Close'][i] > data['R3'][i]:
            data['Signal'][i] = 1
        elif data['Adj Close'][i] < data['S3'][i]:
            data['Signal'][i] = -1
    
    return data['Signal'].iloc[-1]

def sma_andrews_pitchfork_strategy(data):
    # Extract necessary columns
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Compute the midline of the Andrews' Pitchfork using SMA of High-Low over 3 periods
    midline = talib.SMA(high - low, timeperiod=3)
    
    # Compute the upper and lower parallel lines of the Andrews' Pitchfork using the midline and the 
    # highest and lowest price over 3 periods
    upper_line = midline.shift(1) + (high.shift(1) - midline.shift(1))
    lower_line = midline.shift(1) - (midline.shift(1) - low.shift(1))
    
    # Determine the current trend direction based on the position of the close price relative to the midline
    trend = ((close > midline) * 1) - ((close < midline) * 1)
    
    # Determine whether the price has crossed above or below the upper or lower lines
    # and generate buy and sell signals accordingly
    buy_signal = ((trend == 1) & (close > upper_line)) * 1
    sell_signal = ((trend == -1) & (close < lower_line)) * -1
    
    # Combine the buy and sell signals to generate final trading signals
    signals = buy_signal + sell_signal
    
    return signals.iloc[-1]


def sma_ma_support_resistance_strategy(data, period=30):
    """
    Computes the Moving Average Support/Resistance strategy using SMA indicator.

    Parameters:
    data (pandas.DataFrame): A pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    period (int): The time period used to compute the SMA indicator.

    Returns:
    int: A trading signal of 1 (buy) or -1 (sell).

    """

    # Compute the SMA indicator
    data['sma'] = data['Close'].rolling(window=period).mean()

    # Compute the support and resistance levels
    data['sup'] = data['sma'] - data['Close'].rolling(window=period).std()*2
    data['res'] = data['sma'] + data['Close'].rolling(window=period).std()*2

    # Compute the last close price
    last_close = data['Close'].iloc[-1]

    # Compute the last support and resistance levels
    last_sup = data['sup'].iloc[-1]
    last_res = data['res'].iloc[-1]

    # Check if the last close price is below the last support level
    if last_close < last_sup:
        return 1  # buy signal
    # Check if the last close price is above the last resistance level
    elif last_close > last_res:
        return -1  # sell signal
    else:
        return 0  # hold signal

def sma_awesome_oscillator_strategy(data):
    # Load data and calculate SMA and Awesome Oscillator
    df = data
    close_price = df['Adj Close'].values
    sma_fast = df['Adj Close'].rolling(window=5).mean()
    sma_slow = df['Adj Close'].rolling(window=34).mean()
    awesome_oscillator = sma_fast - sma_slow
    
    # Determine trading signals
    signals = []
    prev_ao = None
    for i in range(1, len(df)):
        if awesome_oscillator[i] > 0 and prev_ao is not None and prev_ao <= 0:
            signals.append(1)  # Buy signal
        elif awesome_oscillator[i] < 0 and prev_ao is not None and prev_ao >= 0:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # Hold signal
        prev_ao = awesome_oscillator[i]
        
    # Return trading signals
    return signals[-1]


def ema_macd_strategy(data):
    # Compute the Exponential Moving Averages (EMAs) for 12 and 26 periods
    ema12 = talib.EMA(data['Close'], timeperiod=12)
    ema26 = talib.EMA(data['Close'], timeperiod=26)

    # Compute the MACD line
    macd = ema12 - ema26

    # Compute the Signal line using a 9 period EMA of the MACD line
    signal = talib.EMA(macd, timeperiod=9)

    # Compute the difference between the MACD line and the Signal line
    hist = macd - signal

    # Return 1 if the MACD line crosses above the Signal line, -1 if the MACD line crosses below the Signal line, and 0 otherwise
    if hist[-1] > 0 and hist[-2] < 0:
        return 1
    elif hist[-1] < 0 and hist[-2] > 0:
        return -1
    else:
        return 0


def ema_psar_strategy(df, acceleration_factor=0.02, max_acceleration_factor=0.2):
    """
    EMA Parabolic Stop and Reverse (SAR) strategy.
    Buy when price crosses above PSAR and SAR is below price.
    Sell when price crosses below PSAR and SAR is above price.
    Parameters:
    df (pd.DataFrame): Dataframe containing OHLC data.
    acceleration_factor (float): Acceleration factor used in PSAR calculation (default = 0.02).
    max_acceleration_factor (float): Maximum acceleration factor used in PSAR calculation (default = 0.2).
    Returns:
    int: 1 if a buy signal is generated, -1 if a sell signal is generated, 0 otherwise.
    """
    ema_fast = talib.EMA(df['Close'], timeperiod=12)
    ema_slow = talib.EMA(df['Close'], timeperiod=26)
    psar = talib.SAR(df['High'], df['Low'], acceleration=acceleration_factor, 
                     maximum=max_acceleration_factor)
    
    if df['Close'].iloc[-1] > psar.iloc[-1] and psar.iloc[-1] < df['Low'].iloc[-1] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return 1
    elif df['Close'].iloc[-1] < psar.iloc[-1] and psar.iloc[-1] > df['High'].iloc[-1] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return -1
    else:
        return 0


def ema_ichimoku_strategy(data):
    # Calculate the EMA(26) and EMA(9)
    ema26 = talib.EMA(data['Close'], timeperiod=26)
    ema9 = talib.EMA(data['Close'], timeperiod=9)
    
    # Calculate the Tenkan-sen (Conversion Line) and Kijun-sen (Base Line)
    high9 = talib.MAX(data['High'], timeperiod=9)
    low9 = talib.MIN(data['Low'], timeperiod=9)
    tenkan_sen = (high9 + low9) / 2
    
    high26 = talib.MAX(data['High'], timeperiod=26)
    low26 = talib.MIN(data['Low'], timeperiod=26)
    kijun_sen = (high26 + low26) / 2
    
    # Calculate the Senkou Span A (Leading Span A) and Senkou Span B (Leading Span B)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    high52 = talib.MAX(data['High'], timeperiod=52)
    low52 = talib.MIN(data['Low'], timeperiod=52)
    senkou_span_b = (high52 + low52) / 2
    
    # Calculate the Chikou Span (Lagging Span)
    chikou_span = data['Close'].shift(-26)
    
    # Determine the strategy signal
    signal = 0
    if ema9.iloc[-1] > ema26.iloc[-1] and data['Close'].iloc[-1] > senkou_span_a.iloc[-26] and data['Close'].iloc[-1] > senkou_span_b.iloc[-26] and chikou_span.iloc[-26] > data['Close'].iloc[-1]:
        signal = 1
    elif ema9.iloc[-1] < ema26.iloc[-1] and data['Close'].iloc[-1] < senkou_span_a.iloc[-26] and data['Close'].iloc[-1] < senkou_span_b.iloc[-26] and chikou_span.iloc[-26] < data['Close'].iloc[-1]:
        signal = -1
    
    return signal


def ema_supertrend_strategy(data: pd.DataFrame, n: int = 20, m: float = 3.0) -> int:

    # Calculate EMA
    ema = talib.EMA(data['Close'], timeperiod=n)
    
    # Calculate ATR
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=n)
    
    # Calculate Upper and Lower Bands
    upper_band = ema + m * atr
    lower_band = ema - m * atr
    
    # Calculate Supertrend
    supertrend = pd.Series(0.0, index=data.index)
    for i in range(n, len(data)):
        if data['Close'][i] > upper_band[i - 1]:
            supertrend[i] = lower_band[i]
        elif data['Close'][i] < lower_band[i - 1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i - 1] if data['Close'][i - 1] < supertrend[i - 1] else lower_band[i] if ema[i] < lower_band[i] else upper_band[i]
    
    # Generate trading signal
    if data['Close'][-1] < supertrend[-1]:
        return -1  # Sell signal
    elif data['Close'][-1] > supertrend[-1]:
        return 1  # Buy signal
    else:
        return 0  # Hold signal


def ema_rsi_strategy(data):
    # Load data from CSV and calculate EMA and RSI values
    df = data
    ema = talib.EMA(df['Adj Close'], timeperiod=20)
    rsi = mo.RSIIndicator(df['Adj Close'], window=14).rsi()

    # Apply EMA-RSI strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (rsi.iloc[-1] > 50)
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (rsi.iloc[-1] < 50)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_stochastic_strategy(data):
    # Load data from CSV and calculate EMA and Stochastic values
    df = data
    ema = df['Adj Close'].ewm(span=9, adjust=False).mean()
    stochastic = mo.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Adj Close'], window=14, smooth_window=3).stoch()

    # Apply EMA-Stochastic strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (stochastic.iloc[-1] < 30)
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (stochastic.iloc[-1] > 70)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_cci_strategy(data):
    # Load data from CSV and calculate EMA and CCI values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    cci = tr.cci(df['High'], df['Low'], df['Close'], window=20, constant=0.015)

    # Apply EMA-CCI strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (cci.iloc[-1] > 100)
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (cci.iloc[-1] < -100)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_roc_strategy(data):
    # Load data from CSV and calculate EMA and ROC values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    roc = mo.ROCIndicator(df['Adj Close'], window=14).roc()

    # Apply EMA-ROC strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (roc.iloc[-1] > 0)
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (roc.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_wpr_strategy(data):
    # Load data from CSV and calculate EMA and WPR values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    wpr = mo.WilliamsRIndicator(df['High'], df['Low'], df['Close'], 14)

    # Apply EMA-WPR strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (wpr.williams_r()[-1] > -80)
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (wpr.williams_r()[-1] < -20)	

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_macd_hist_strategy(data):
    # Load data from CSV and calculate EMA and MACD values
    df = data
    ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    # Apply EMA-MACD Hist strategy
    buy_signal = (macd_hist.iloc[-2] < 0) and (macd_hist.iloc[-1] > 0)
    sell_signal = (macd_hist.iloc[-2] > 0) and (macd_hist.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_bbands_strategy(data):
    # Load data from CSV and calculate EMA and Bollinger Bands values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    std = df['Adj Close'].rolling(window=20).std()
    upper_band = ema + 2 * std
    lower_band = ema - 2 * std

    # Apply EMA-Bollinger Bands strategy
    buy_signal = (df['Adj Close'].iloc[-1] < lower_band.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] > upper_band.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_atr_strategy(data):
    # Load data from CSV and calculate EMA and ATR values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    atr = vo.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)

    # Apply EMA-ATR strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (df['Low'].iloc[-1] > (ema.iloc[-1] - atr.average_true_range()[-1]))
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (df['High'].iloc[-1] < (ema.iloc[-1] + atr.average_true_range()[-1]))	

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_stdev_strategy(data):
    # Load data from CSV and calculate EMA and STDEV values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    stdev = df['Adj Close'].rolling(window=20).std()

    # Apply EMA-STDEV strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1] + stdev.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1] - stdev.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_kc_strategy(data):
    # Load data from CSV and calculate EMA and Keltner Channels values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=10)
    upper_kc = ema + 2 * atr
    lower_kc = ema - 2 * atr

    # Apply EMA-KC strategy
    buy_signal = (df['Adj Close'].iloc[-1] > upper_kc.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < lower_kc.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_donchian_strategy(data):
    # Load data from CSV and calculate EMA and Donchian Channel values
    df = data
    ema_fast = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Adj Close'].ewm(span=26, adjust=False).mean()
    donchian_upper = df['Adj Close'].rolling(window=20).max()
    donchian_lower = df['Adj Close'].rolling(window=20).min()

    # Apply EMA-Donchian strategy
    buy_signal = (ema_fast.iloc[-1] > ema_slow.iloc[-1]) and (df['Adj Close'].iloc[-1] > donchian_upper.iloc[-1])
    sell_signal = (ema_fast.iloc[-1] < ema_slow.iloc[-1]) or (df['Adj Close'].iloc[-1] < donchian_lower.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_chandelier_exit_strategy(data):
    # Load data from CSV and calculate EMA and ATR values
    df = data
    df.columns = [x.title() for x in df.columns]
    # print(df)
    ema = df['Adj Close'].ewm(span=22, adjust=False).mean()
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=22)

    # Calculate Chandelier Exit long and short values
    long_chandelier_exit = ema - 3 * atr
    short_chandelier_exit = ema + 3 * atr

    # Determine signal for last OHLCV row
    if df['Adj Close'].iloc[-1] > long_chandelier_exit.iloc[-1]:
        return 1
    elif df['Adj Close'].iloc[-1] < short_chandelier_exit.iloc[-1]:
        return -1
    else:
        return 0


def ema_obv_strategy(data):
    # Load data from CSV and calculate EMA and OBV values
    df = data
    ema = tr.ema_indicator(df['Adj Close'], window=20)
    obv = [0]
    for i in range(1, len(df)):
        if df['Adj Close'][i] > df['Adj Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Adj Close'][i] < df['Adj Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
	
    # Apply EMA-OBV strategy
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (df['OBV'].iloc[-1] > df['OBV'].iloc[-2])
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) or (df['OBV'].iloc[-1] < df['OBV'].iloc[-2])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_cmf_strategy(data):
    # Load data from CSV and calculate EMA and CMF values
    df = data
    ema_fast = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Adj Close'].ewm(span=26, adjust=False).mean()
    cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()

    # Apply EMA-CMF strategy
    buy_signal = (ema_fast.iloc[-1] > ema_slow.iloc[-1]) and (cmf.iloc[-1] > 0)
    sell_signal = (ema_fast.iloc[-1] < ema_slow.iloc[-1]) or (cmf.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_vroc_strategy(data):
    # Load data from CSV and calculate EMA and VROC values
    df = data
    ema_short = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_long = df['Adj Close'].ewm(span=26, adjust=False).mean()
    vroc = mo.ROCIndicator(df['Volume'], window=14).roc() / 100

    # Apply EMA-VROC strategy
    buy_signal = (ema_short.iloc[-1] > ema_long.iloc[-1]) and (vroc.iloc[-1] > 0)
    sell_signal = (ema_short.iloc[-1] < ema_long.iloc[-1]) or (vroc.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_mfi_strategy(data):
    # Load data from CSV and calculate EMA and MFI values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    mfi = MFIIndicator(df['High'], df['Low'], df['Adj Close'], df['Volume'], window=14).money_flow_index()

    # Apply EMA-MFI strategy
    buy_signal = (ema.iloc[-1] > ema.iloc[-2]) and (mfi.iloc[-1] < 20)
    sell_signal = (ema.iloc[-1] < ema.iloc[-2]) or (mfi.iloc[-1] > 80)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_adl_strategy(data):
    # Load data from CSV and calculate ADL and EMA values
    df = data
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    adl = money_flow_volume.cumsum()
    ema_adl = adl.ewm(span=20, adjust=False).mean()

    # Apply EMA-ADL strategy
    buy_signal = (df['Close'].iloc[-1] > ema_adl.iloc[-1])
    sell_signal = (df['Close'].iloc[-1] < ema_adl.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_eom_strategy(data):
    # Load data from CSV and calculate EMA and EOM values
    df = data
    ema = tr.ema_indicator(df['Close'], window=20)
    eom = EaseOfMovementIndicator(
        high=df['High'], low=df['Low'], volume=df['Volume'], window=14, fillna=False
    ).sma_ease_of_movement()

    # Apply EMA-EOM strategy
    buy_signal = (ema.iloc[-1] > ema.iloc[-2]) and (eom.iloc[-1] > 0)
    sell_signal = (ema.iloc[-1] < ema.iloc[-2]) or (eom.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def ema_pivot_points_strategy(data):
    # Load data from CSV and calculate EMA and Pivot Point levels
    df = data
    ema_short = df['Adj Close'].ewm(span=20, adjust=False).mean()
    ema_long = df['Adj Close'].ewm(span=50, adjust=False).mean()
    pivot = (df['High'] + df['Low'] + df['Adj Close']) / 3
    r1 = 2 * pivot - df['Low']
    s1 = 2 * pivot - df['High']
    r2 = pivot + (df['High'] - df['Low'])
    s2 = pivot - (df['High'] - df['Low'])
    r3 = df['High'] + 2 * (pivot - df['Low'])
    s3 = df['Low'] - 2 * (df['High'] - pivot)

    # Apply EMA-Pivot Point strategy
    buy_signal = (ema_short.iloc[-1] > ema_long.iloc[-1]) and (df['Adj Close'].iloc[-1] > r1.iloc[-1])
    sell_signal = (ema_short.iloc[-1] < ema_long.iloc[-1]) and (df['Adj Close'].iloc[-1] < s1.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_fibonacci_strategy(data):
    # Load data from CSV and calculate EMA and Fibonacci retracement levels
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    pivot_high = high.rolling(window=10, center=False).max()
    pivot_low = low.rolling(window=10, center=False).min()
    diff = pivot_high - pivot_low
    levels = [0.236, 0.382, 0.5, 0.618, 0.764]

    # Apply EMA-Fibonacci retracement strategy
    buy_signal = (close.iloc[-1] > ema.iloc[-1]) and (close.iloc[-1] <= (pivot_low.iloc[-1] + levels[0] * diff.iloc[-1]))
    sell_signal = (close.iloc[-1] < ema.iloc[-1]) or (close.iloc[-1] >= (pivot_high.iloc[-1] - levels[0] * diff.iloc[-1]))

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

    
def ema_srl_strategy(data, sma_periods=[20, 50]):
    # Compute SMAs
    for period in sma_periods:
        sma_label = f"EMA_{period}"
        data[sma_label] = data['Close'].ewm(period).mean()

    # Compute potential support and resistance levels
    data['SRL'] = 0
    for i in range(max(sma_periods), len(data)):
        if data['Close'][i] < data['EMA_%d' % sma_periods[0]][i] and data['Close'][i] > data['EMA_%d' % sma_periods[1]][i]:
            data['SRL'][i] = 1  # Resistance level
        elif data['Close'][i] > data['EMA_%d' % sma_periods[0]][i] and data['Close'][i] < data['EMA_%d' % sma_periods[1]][i]:
            data['SRL'][i] = -1  # Support level
	
	# data['SRL'].fillna(0, inplace=True)
    # Return last signal
    return int(data['SRL'][-1])

def ema_gann_lines_strategy(data):
    # Load data from CSV and calculate EMA values
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()

    # Determine trend direction using EMA values
    trend_up = ema.iloc[-1] > ema.iloc[-2]

    # Draw Gann Lines on the price chart based on trend direction
    if trend_up:
        resistance = df['Adj Close'].iloc[-1] + 0.25 * (df['Adj Close'].iloc[-1] - ema.iloc[-1])
        support = df['Adj Close'].iloc[-1] - 0.25 * (df['Adj Close'].iloc[-1] - ema.iloc[-1])
    else:
        resistance = df['Adj Close'].iloc[-1] - 0.25 * (ema.iloc[-1] - df['Adj Close'].iloc[-1])
        support = df['Adj Close'].iloc[-1] + 0.25 * (ema.iloc[-1] - df['Adj Close'].iloc[-1])

    # Apply Gann Lines strategy
    buy_signal = (df['Adj Close'].iloc[-1] >= support) and (ema.iloc[-1] >= support)
    sell_signal = (df['Adj Close'].iloc[-1] <= resistance) and (ema.iloc[-1] <= resistance)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_andrews_pitchfork_strategy(data):
    # Load data from CSV and calculate EMA and Andrews Pitchfork
    df = data
    ema = df['Adj Close'].ewm(span=20, adjust=False).mean()
    highs = df['High']
    lows = df['Low']
    pivot = (highs + lows + ema) / 3
    diff = highs - lows
    support1 = pivot - (0.3333 * diff)
    resistance1 = pivot + (0.3333 * diff)
    support2 = pivot - (0.6666 * diff)
    resistance2 = pivot + (0.6666 * diff)
    support3 = pivot - diff
    resistance3 = pivot + diff

    # Apply EMA-Andrews Pitchfork strategy
    buy_signal = (df['Adj Close'].iloc[-1] > resistance1.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < support1.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_sr_strategy(data):
    # Load data from CSV and calculate EMA and support/resistance levels
    df = data
    ema = df['Adj Close'].ewm(span=20).mean()
    support_levels = df['Adj Close'].rolling(window=20).min()
    resistance_levels = df['Adj Close'].rolling(window=20).max()

    # Determine buy and sell signals based on EMA and support/resistance levels
    buy_signal = (df['Adj Close'].iloc[-1] > ema.iloc[-1]) and (df['Adj Close'].iloc[-1] > support_levels.iloc[-1])
    sell_signal = (df['Adj Close'].iloc[-1] < ema.iloc[-1]) and (df['Adj Close'].iloc[-1] < resistance_levels.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def ema_awesome_oscillator_strategy(data):
    # Load data from CSV and calculate EMA and Awesome Oscillator values
    df = data
    ema_fast = df['Adj Close'].ewm(span=5, adjust=False).mean()
    ema_slow = df['Adj Close'].ewm(span=34, adjust=False).mean()
    awesome_oscillator = ema_fast - ema_slow

    # Apply EMA-Awesome Oscillator strategy
    buy_signal = (awesome_oscillator.iloc[-1] > 0) and (awesome_oscillator.iloc[-2] < 0)
    sell_signal = (awesome_oscillator.iloc[-1] < 0) and (awesome_oscillator.iloc[-2] > 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

# def macd_strategy(data):
#     # Load data from CSV and calculate MACD values
#     df = data
#     ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
#     ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
#     macd_line = ema_12 - ema_26
#     signal_line = macd_line.ewm(span=9, adjust=False).mean()
#     macd_histogram = macd_line - signal_line

#     # Apply MACD strategy
#     buy_signal = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
#     sell_signal = (macd_line.iloc[-1] < signal_line.iloc[-1]) and (macd_line.iloc[-2] >= signal_line.iloc[-2])

#     # Determine signal for last OHLCV row
#     if buy_signal:
#         return 1
#     elif sell_signal:
#         return -1
#     else:
#         return 0
def macd_strategy(data):
    # Extract the close price from the data
    close = data['Close']

    # Calculate MACD with period (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # Calculate MACD with period (5, 10, 3)
    macd2, macd_signal2, macd_hist2 = talib.MACD(close, fastperiod=5, slowperiod=10, signalperiod=3)

    # Determine signal for last OHLCV row
    if macd[-1] > macd_signal[-1] and macd2[-1] > macd_signal2[-1]:
        return 1
    elif macd[-1] < macd_signal[-1] and macd2[-1] < macd_signal2[-1]:
        return -1
    else:
        return 0

def macd_psar_strategy(data):
    # Load data from CSV and calculate MACD and Parabolic SAR values
    df = data
    macd = tr.MACD(df['Adj Close'], window_fast=12, window_slow=26, window_sign=9)
    psar = tr.PSARIndicator(df['High'], df['Low'], df['Adj Close'], step=0.02, max_step=0.2).psar()
	
    # Apply MACD-PSAR strategy
    buy_signal = (macd.macd_signal().iloc[-1] > 0) and (df['Adj Close'].iloc[-1] > psar.iloc[-1])
    sell_signal = (macd.macd_signal().iloc[-1] < 0) or (df['Adj Close'].iloc[-1] < psar.iloc[-1])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_ichimoku_strategy(df):
    # Compute MACD and Ichimoku indicators
    macd = tr.MACD(df["Close"]).macd()
    ichimoku = tr.IchimokuIndicator(df["High"], df["Low"], 9, 26, 52).ichimoku_conversion_line()
    ichimoku_base = tr.IchimokuIndicator(df["High"], df["Low"], 9, 26, 52).ichimoku_base_line()

    # Determine signals based on MACD and Ichimoku indicators
    buy_signal = (macd > ichimoku) & (df["Close"] > ichimoku_base)
    sell_signal = (macd < ichimoku) & (df["Close"] < ichimoku_base)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def macd_supertrend_atr_strategy(data, macd_fast=12, macd_slow=26, macd_signal=9, supertrend_multiplier=3, atr_window=14):
    # Calculate MACD
    ema_fast = data['Close'].ewm(span=macd_fast, min_periods=macd_fast).mean()
    ema_slow = data['Close'].ewm(span=macd_slow, min_periods=macd_slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, min_periods=macd_signal).mean()
    macd_hist = macd - signal

    # Calculate Supertrend
    atr = data['High'].sub(data['Low']).abs().rolling(atr_window).mean()
    basic_upper_band = (data['High'] + data['Low']) / 2 + supertrend_multiplier * atr
    basic_lower_band = (data['High'] + data['Low']) / 2 - supertrend_multiplier * atr
    basic_trend = np.where(data['Close'] > basic_upper_band.shift(), 1, np.where(data['Close'] < basic_lower_band.shift(), -1, 0))
    supertrend = np.where(basic_trend == 0, np.where(basic_upper_band.shift() < basic_upper_band, basic_upper_band, basic_upper_band.shift()), np.where(basic_trend > 0, basic_lower_band, basic_upper_band))

    # Determine signal for last OHLCV row
    if supertrend[-1] == 1 and macd_hist[-1] > 0:
        return 1
    elif supertrend[-1] == -1 and macd_hist[-1] < 0:
        return -1
    else:
        return 0


import pandas as pd
import numpy as np
import talib

def macd_rsi_strategy(df, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14, rsi_upper=70, rsi_lower=30):
    # Compute MACD and signal lines
    macd, macd_signal_line, _ = talib.MACD(df['Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)

    # Compute RSI
    rsi = talib.RSI(df['Close'], timeperiod=rsi_period)

    # Compute signal
    buy_signal = (macd > macd_signal_line) & (rsi < rsi_lower)
    sell_signal = (macd < macd_signal_line) & (rsi > rsi_upper)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def macd_stochastic_strategy(data):
    # Initialize indicators
    macd = MACD(data['Close']).macd()
    macd_signal = MACD(data['Close']).macd_signal()
    stoch = StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    
    # Determine signals
    buy_signal = (macd > macd_signal) & (stoch < 20)
    sell_signal = (macd < macd_signal) & (stoch > 80)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def macd_cci_strategy(df, fast_period=12, slow_period=26, signal_period=9, cci_period=14, cci_upper=100, cci_lower=-100):
    # Calculate MACD
    macd, signal, hist = talib.MACD(df['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    # Calculate CCI
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=cci_period)
    
    # Determine buy and sell signals based on MACD and CCI
    buy_signal = (macd > signal) & (cci < cci_lower)
    sell_signal = (macd < signal) & (cci > cci_upper)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_roc_strategy(df, n_fast=12, n_slow=26, n_signal=9, n_roc=10):
    # Calculate MACD
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=n_fast, slowperiod=n_slow, signalperiod=n_signal)
    
    # Calculate ROC
    roc = talib.ROC(df['Close'], timeperiod=n_roc)
    
    # Determine if MACD and ROC are above or below zero line
    macd_above_zero = macdhist > 0
    roc_above_zero = roc > 0
    
    # Determine if MACD and ROC are bullish or bearish
    macd_bullish = macdhist > macdhist.shift(1)
    roc_bullish = roc > roc.shift(1)
    
    # Determine buy and sell signals
    buy_signal = macd_above_zero & roc_above_zero & macd_bullish & roc_bullish
    sell_signal = (~macd_above_zero) & (~roc_above_zero) & (~macd_bullish) & (~roc_bullish)
    
    # Determine signal for last row of OHLC data
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_wpr_strategy(df, fast_period=12, slow_period=26, signal_period=9, wpr_period=14, wpr_oversold=-80, wpr_overbought=-20):
    # Compute MACD
    # macd = tr.MACD(df['Close'], fast_period, slow_period, signal_period)
    # df['macd'] = macd.macd()
    # df['signal'] = macd.macd_signal()
	macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=slow_period)

    # Compute Williams %R (WPR)
	wpr = mo.WilliamsRIndicator(df['High'], df['Low'], df['Close'], wpr_period)
	df['wpr'] = wpr.williams_r()
    
    # Determine buy and sell signals
    # buy_signal = (df['macd'] > df['signal']) & (df['wpr'] < wpr_oversold)
    # sell_signal = (df['macd'] < df['signal']) & (df['wpr'] > wpr_overbought)
	buy_signal = (macd > macdsignal) & (df['wpr'][-1] < wpr_oversold)
	sell_signal = (macd < macdsignal) & (df['wpr'][-1] > wpr_overbought)
    
	# Determine signal for last OHLCV row
	if buy_signal.iloc[-1]:
		return 1
	elif sell_signal.iloc[-1]:
		return -1
	else:
		return 0

def macd_hist_strategy(data, fast_period=12, slow_period=26, signal_period=9):
    # Compute MACD line and signal line
    macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    # Compute EMA crossover signals
    ema_fast = talib.EMA(data['Close'], timeperiod=fast_period)
    ema_slow = talib.EMA(data['Close'], timeperiod=slow_period)
    ema_signal = np.where(ema_fast > ema_slow, 1, -1)
    
    # Compute MACD histogram signals
    macd_signal = np.where(macdhist > 0, 1, -1)
    
    # Combine signals
    buy_signal = (ema_signal[-1] == 1) and (macd_signal[-1] == 1)
    sell_signal = (ema_signal[-1] == -1) and (macd_signal[-1] == -1)
    
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_bbands_strategy(df, n_fast=12, n_slow=26, n_signal=9, n_bbands=20, dev=2):
    # Calculate MACD
    macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=n_fast, slowperiod=n_slow, signalperiod=n_signal)
    
    # Calculate Bollinger Bands
    middle_band = df['Close'].rolling(window=n_bbands).mean()
    upper_band = middle_band + dev * df['Close'].rolling(window=n_bbands).std()
    lower_band = middle_band - dev * df['Close'].rolling(window=n_bbands).std()
    
    # Determine buy/sell signals based on MACD and Bollinger Bands
    buy_signal = (macd > macd_signal) & (df['Close'] < lower_band)
    sell_signal = (macd < macd_signal) & (df['Close'] > upper_band)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_atr_strategy(data, fast_period=12, slow_period=26, signal_period=9, atr_period=14, multiplier=2):
    # Calculate MACD
    macd, macd_signal, macd_hist = talib.MACD(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)

    # Calculate ATR
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_period)

    # Calculate upper and lower bands
    upper_band = data['Close'] + (multiplier * atr)
    lower_band = data['Close'] - (multiplier * atr)

    # Determine buy and sell signals
    buy_signal = (macd_hist > 0) & (data['Close'] > upper_band)
    sell_signal = (macd_hist < 0) & (data['Close'] < lower_band)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_stdev_strategy(df, n_fast=12, n_slow=26, n_signal=9, n_stdev=20):
    # Compute MACD
    macd, macd_signal, _ = talib.MACD(df['Close'], n_fast, n_slow, n_signal)
    
    # Compute standard deviation
    stdev = df['Close'].rolling(window=n_stdev).std()

    # Compute buy/sell signals
    buy_signal = (df['Close'] > stdev) & (macd > macd_signal)
    sell_signal = (df['Close'] <= stdev) & (macd < macd_signal)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_kc_strategy(df):
    # Calculate MACD indicators
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # Calculate Keltner Channels
    kc_mid = df['Close'].ewm(span=20, adjust=False).mean()
    kc_range = df['High'] - df['Low']
    kc_atr = kc_range.ewm(span=2, adjust=False).mean()
    kc_upper = kc_mid + 2 * kc_atr
    kc_lower = kc_mid - 2 * kc_atr
    
    # Determine buy/sell/hold signals
    buy_signal = (macd_line > signal_line) & (df['Close'] > kc_upper)
    sell_signal = (macd_line < signal_line) & (df['Close'] < kc_lower)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_donchian_strategy(df, fast_period=12, slow_period=26, signal_period=9, donchian_period=20):
    # Calculate MACD
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()

    # Calculate Donchian Channel
    df['High_rolling'] = df['High'].rolling(window=donchian_period).max()
    df['Low_rolling'] = df['Low'].rolling(window=donchian_period).min()
    df['DC_upper'] = df['High_rolling'].shift(1)
    df['DC_lower'] = df['Low_rolling'].shift(1)

    # Determine buy and sell signals
    buy_signal = (df['Close'].shift(1) > df['DC_upper'].shift(1)) & (df['Close'] < df['DC_upper'])
    sell_signal = (df['Close'].shift(1) < df['DC_lower'].shift(1)) & (df['Close'] > df['DC_lower'])

    # Determine MACD signal
    macd_signal = macd - signal
    macd_signal = np.where(macd_signal > 0, 1, -1)

    # Combine signals
    signals = np.zeros(len(df))
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    signals = signals + macd_signal
    signals = np.where(signals > 1, 1, np.where(signals < -1, -1, signals))

    return int(signals[-1])


# def macd_chandelier_exit_strategy(df, period=22, atr_multiplier=3):
#     # Calculate MACD
#     macd, signal, hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#     df['MACD'] = macd
#     df['Signal'] = signal
#     df['MACD_hist'] = hist

#     # Calculate ATR
#     df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)

#     # Calculate Chandelier Exit
#     long_stop = df['High'].rolling(period).max() - atr_multiplier * df['ATR']
#     short_stop = df['Low'].rolling(period).min() + atr_multiplier * df['ATR']

#     # Determine signal for last OHLCV row
#     if df['Close'].iloc[-1] > long_stop.iloc[-1]:
#         return 1
#     elif df['Close'].iloc[-1] < short_stop.iloc[-1]:
#         return -1
#     else:
#         return 0

def macd_chandelier_exit_strategy(data):
    # Load data from CSV and calculate MACD and Chandelier Exit values
    df = data
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    atr = df['High'] - df['Low']
    atr = atr.rolling(window=14).mean()
    chandelier_exit = df['High'].rolling(window=22).max() - atr * 3

    # Apply MACD-Chandelier Exit strategy
    buy_signal = macd.iloc[-1] > signal.iloc[-1] and df['Adj Close'].iloc[-1] > chandelier_exit.iloc[-1]
    sell_signal = macd.iloc[-1] < signal.iloc[-1] or df['Adj Close'].iloc[-1] < chandelier_exit.iloc[-1]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_obv_strategy(data):
    # Load data from CSV and calculate MACD and OBV values
    df = data
    df['macd'], _, _ = talib.MACD(df['Adj Close'])
    obv = [0]
    for i in range(1, len(df)):
        if df['Adj Close'][i] > df['Adj Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Adj Close'][i] < df['Adj Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv    
    # sdf['obv'] = obv#ta.obv(df['close'], df['volume'])

    # Apply MACD-OBV strategy
    buy_signal = (df['macd'].iloc[-1] > 0) & (df['obv'].iloc[-1] > df['obv'].iloc[-2])
    sell_signal = (df['macd'].iloc[-1] < 0) & (df['obv'].iloc[-1] < df['obv'].iloc[-2])

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_cmf_strategy(data):
    # Load data from CSV and calculate MACD and CMF values
    df = data
    stock = Sdf.retype(df)
    # stock.drop_duplicates()
    macd = stock['macd']
    signal = stock['macds']
    cmf = stock['cmf']

    # Apply MACD-CMF strategy
    buy_signal = (macd.iloc[-1] > signal.iloc[-1]) & (cmf.iloc[-1] > 0)
    sell_signal = (macd.iloc[-1] < signal.iloc[-1]) & (cmf.iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_cmf_strategy(data):
    # Calculate MACD and CMF
    df = data.copy()
    df['ema12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['cmf'] = ((2 * df['Adj Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])).rolling(window=20).mean()

    # Determine signals
    buy_signal = (df['macd'].iloc[-1] > 0) & (df['cmf'].iloc[-1] > 0)
    sell_signal = (df['macd'].iloc[-1] < 0) & (df['cmf'].iloc[-1] < 0)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_vroc_strategy(data):
    # Load data from CSV
    df = data
    
    # Calculate MACD and VROC using talib
    macd, macd_signal, macd_hist = talib.MACD(df['Adj Close'])
    vroc = talib.ROC(df['Volume'], timeperiod=14)
    # print(vroc)
    # Determine signals based on MACD and VROC
    if macd[-1] > macd_signal[-1] and vroc[-1] > 0:
        return 1  # Buy signal
    elif macd[-1] < macd_signal[-1] and vroc[-1] < 0:
        return -1  # Sell signal
    else:
        return 0  # Hold signal


def macd_mfi_strategy(data):
    # Load data from CSV and calculate MACD and MFI values
    df = data
    close = df['Adj Close']
    macd, macdsignal, macdhist = talib.MACD(close)
    mfi = talib.MFI(df['High'], df['Low'], close, df['Volume'], timeperiod=14)

    # Apply MACD-MFI strategy
    buy_signal = (macdhist[-2] < 0) and (macdhist[-1] > 0) and (mfi[-1] < 30)
    sell_signal = (macdhist[-2] > 0) and (macdhist[-1] < 0) and (mfi[-1] > 70)

    # Determine signal for last row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_adl_strategy(data):
    # Load data from CSV
    df = data

    # Calculate ADL
    adl = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])

    # Calculate MACD
    macd, signal, hist = talib.MACD(df['Close'])

    # Apply MACD-ADL strategy
    buy_signal = (macd[-1] > signal[-1]) and (adl[-1] > adl[-2])
    sell_signal = (macd[-1] < signal[-1]) and (adl[-1] < adl[-2])

    # Determine signal for last row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_eom_strategy(data):
    # Load data from CSV and calculate EOM and MACD values
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    distance_moved = ((high + low) / 2) - talib.SMA((high + low) / 2, timeperiod=14)
    box_ratio = (volume / 100000000) / (high - low)
    one_day_eom = distance_moved / box_ratio
    eom = talib.SMA(one_day_eom, timeperiod=14) / talib.SMA(volume, timeperiod=14)

    # Calculate MACD values
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # Apply MACD-EOM strategy
    buy_signal = macd[-1] > signal[-1] and eom[-1] > eom[-2]
    sell_signal = macd[-1] < signal[-1] and eom[-1] < eom[-2]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def macd_pivot_points_strategy(data):
    # Load data from CSV and calculate MACD and pivot points
    df = data.copy()
    df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['pivot_shift'] = df['pivot'].shift(1)
    df['high_shift'] = df['High'].shift(1)
    df['low_shift'] = df['Low'].shift(1)
    df['close_shift'] = df['Close'].shift(1)
    df['r1'] = (2 * df['pivot']) - df['Low']
    df['s1'] = (2 * df['pivot']) - df['High']
    df['r2'] = df['pivot'] + (df['high_shift'] - df['low_shift'])
    df['s2'] = df['pivot'] - (df['high_shift'] - df['low_shift'])
    df['r3'] = df['high_shift'] + 2 * (df['pivot'] - df['low_shift'])
    df['s3'] = df['low_shift'] - 2 * (df['high_shift'] - df['pivot'])

    macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Apply MACD-Pivot Points strategy
    buy_signal = (macd[-1] > signal[-1]) and (df['Close'].iloc[-1] > df['pivot_shift'][-1])
    sell_signal = (macd[-1] < signal[-1]) and (df['Close'].iloc[-1] < df['pivot_shift'][-1])
    
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_fibonacci_strategy(data):
    # Load data from CSV and calculate MACD values
    df = data
    close_prices = df['Adj Close'].values
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

    # Calculate Fibonacci retracement levels
    recent_high = df['High'].max()
    recent_low = df['Low'].min()
    vertical_distance = recent_high - recent_low
    fib236 = recent_high - vertical_distance * 0.236
    fib382 = recent_high - vertical_distance * 0.382
    fib50 = recent_high - vertical_distance * 0.5
    fib618 = recent_high - vertical_distance * 0.618
    fib100 = recent_high - vertical_distance

    # Identify potential support and resistance levels based on MACD and Fibonacci
    buy_signal = (macd > macdsignal) & (df['Adj Close'] <= fib236)
    sell_signal = (macd < macdsignal) & (df['Adj Close'] >= fib618)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_srl_strategy(data):
    # Load data from CSV and calculate MACD and EMA values
    df = data
    close = df['Adj Close']
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()
    macd, macd_signal, macd_hist = talib.MACD(close)

    # Calculate support and resistance levels
    resistance_level = max(ema_10[-1], ema_30[-1])
    support_level = min(ema_10[-1], ema_30[-1])

    # Apply SRL strategy
    buy_signal = macd[-1] > macd_signal[-1] and close[-1] > resistance_level
    sell_signal = macd[-1] < macd_signal[-1] and close[-1] < support_level

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_gann_lines_strategy(data):
    # Load data from CSV and calculate MACD
    df = data
    close = df['Adj Close']
    macd, macd_signal, macd_hist = talib.MACD(close)

    # Calculate Gann Lines manually
    high = df['High'].max()
    low = df['Low'].min()
    range_ = high - low
    levels = []
    for i in range(1, 9):
        level = high - (range_ * i / 8)
        levels.append(level)

    # Check if MACD is above or below each Gann Line
    signals = []
    for i in range(8):
        if macd[-1] > levels[i]:
            signals.append(1)
        else:
            signals.append(-1)

    # Determine signal for last row
    if sum(signals) > 0:
        return 1
    elif sum(signals) < 0:
        return -1
    else:
        return 0
    

def macd_andrews_pitchfork_strategy(data):
    # Get the MACD indicator
    macd, _, _ = talib.MACD(data['Close'])
    
    # Get the Andrews Pitchfork levels manually
    high = data['High']
    low = data['Low']
    median_price = (high + low) / 2.0
    x = len(data)
    af_upper = np.zeros(x)
    af_lower = np.zeros(x)
    
    for i in range(1, x):
        p1 = [i - 3, low[i - 3]]
        p2 = [i - 1, high[i - 1]]
        p3 = [i, high[i]]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p2[1] - (slope * p2[0])
        af_upper[i] = (slope * p3[0]) + intercept
        
        p1 = [i - 3, high[i - 3]]
        p2 = [i - 1, low[i - 1]]
        p3 = [i, low[i]]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p2[1] - (slope * p2[0])
        af_lower[i] = (slope * p3[0]) + intercept
    
    # Determine buy/sell signals
    buy_signal = False
    sell_signal = False
    
    for i in range(2, x):
        if macd[i-2] < macd[i-1] and macd[i-1] > macd[i]:
            if af_lower[i-1] < median_price[i] and af_lower[i] >= median_price[i]:
                buy_signal = True
        elif macd[i-2] > macd[i-1] and macd[i-1] < macd[i]:
            if af_upper[i-1] > median_price[i] and af_upper[i] <= median_price[i]:
                sell_signal = True
    
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def macd_ma_sr_strategy(data):
    # Extract required columns from data
    close = data['Close']
    
    # Calculate MACD
    macd, macd_signal, _ = talib.MACD(close)
    
    # Calculate moving averages
    sma_10 = talib.SMA(close, timeperiod=10)
    sma_50 = talib.SMA(close, timeperiod=50)
    
    # Calculate support and resistance levels
    resistance = pd.concat([sma_10, sma_50]).groupby(level=0).max()
    support = pd.concat([sma_10, sma_50]).groupby(level=0).min()
    
    # Determine buy/sell signals
    buy_signal = (macd > macd_signal) & (close > resistance)
    sell_signal = (macd < macd_signal) & (close < support)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def macd_awesome_oscillator_strategy(data):
    # Define inputs
    close = data['Close']
    
    # Calculate MACD
    macd, macdsignal, macdhist = talib.MACD(close)
    
    # Calculate Awesome Oscillator
    median_price = (data['High'] + data['Low']) / 2
    ao_short = talib.SMA(median_price, timeperiod=5)
    ao_long = talib.SMA(median_price, timeperiod=34)
    awesome_oscillator = ao_short - ao_long
    
    # Determine signal for last row
    if macdhist.iloc[-1] > 0 and awesome_oscillator.iloc[-1] > 0:
        return 1
    elif macdhist.iloc[-1] < 0 and awesome_oscillator.iloc[-1] < 0:
        return -1
    else:
        return 0

def psar_strategy(data):
    psar1 = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    psar2 = talib.SAR(data['High'], data['Low'], acceleration=0.01, maximum=0.1)
    
    buy_signal = (data['Close'] > psar1) & (data['Close'] > psar2)
    sell_signal = (data['Close'] < psar1) | (data['Close'] < psar2)
    
    if buy_signal.any():
        return 1
    elif sell_signal.any():
        return -1
    else:
        return 0


def psar_ichimoku_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate Ichimoku Cloud
    high = data['High']
    low = data['Low']
    conversion_period = 9
    base_period = 26
    span_period = 52
    displacement = 26
    conversion_line = talib.EMA(high, timeperiod=conversion_period) + talib.EMA(low, timeperiod=conversion_period)
    base_line = talib.EMA(high, timeperiod=base_period) + talib.EMA(low, timeperiod=base_period)
    span_a = (conversion_line + base_line) / 2
    span_b = talib.EMA(high, timeperiod=span_period) + talib.EMA(low, timeperiod=span_period)
    span_a = span_a.shift(displacement)
    span_b = span_b.shift(displacement)
    tenkan_sen = (talib.MAX(high, timeperiod=conversion_period) + talib.MIN(low, timeperiod=conversion_period)) / 2
    kijun_sen = (talib.MAX(high, timeperiod=base_period) + talib.MIN(low, timeperiod=base_period)) / 2
    
    # Determine signals
    buy_signal = False
    sell_signal = False
    
    if psar.iloc[-1] < data['Close'].iloc[-1]:
        if data['Close'].iloc[-1] > span_a.iloc[-1] and data['Close'].iloc[-1] > span_b.iloc[-1]:
            if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1] and tenkan_sen.iloc[-2] <= kijun_sen.iloc[-2]:
                buy_signal = True
    elif psar.iloc[-1] > data['Close'].iloc[-1]:
        if data['Close'].iloc[-1] < span_a.iloc[-1] and data['Close'].iloc[-1] < span_b.iloc[-1]:
            if tenkan_sen.iloc[-1] < kijun_sen.iloc[-1] and tenkan_sen.iloc[-2] >= kijun_sen.iloc[-2]:
                sell_signal = True
                
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0

def psar_supertrend_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate SuperTrend
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=10)
    upper_band = (data['High'] + data['Low']) / 2 + (3 * atr)
    lower_band = (data['High'] + data['Low']) / 2 - (3 * atr)
    super_trend = np.where(data['Close'] > lower_band, 1, -1)
    for i in range(1, len(data)):
        if super_trend[i-1] == 1 and lower_band[i] > psar[i]:
            super_trend[i] = 1
        elif super_trend[i-1] == -1 and upper_band[i] < psar[i]:
            super_trend[i] = -1
    
    # Determine signal for last OHLCV row
    buy_signal = (super_trend[-1] == 1)
    sell_signal = (super_trend[-1] == -1)
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_rsi_strategy(data, psar_acceleration=0.02, psar_maximum=0.2, rsi_period=14, rsi_buy_threshold=30, rsi_sell_threshold=70):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=psar_acceleration, maximum=psar_maximum)

    # Calculate RSI
    rsi = talib.RSI(data['Close'], timeperiod=rsi_period)

    # Determine buy and sell signals
    buy_signal = (rsi < rsi_buy_threshold) & (data['Close'] > psar)
    sell_signal = (rsi > rsi_sell_threshold) | (data['Close'] < psar)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_stochastic_strategy(data):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate PSAR
    psar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # Calculate Stochastic
    # stoch = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    stoch = mo.StochasticOscillator(high, low, close, window=14).stoch()

    # Determine buy and sell signals
    buy_signal = (psar[-1] < close[-1]) and (stoch[-1] < 20)
    sell_signal = (psar[-1] > close[-1]) and (stoch[-1] > 80)

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_cci_strategy(data):
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values

    # Calculate PSAR with default parameters
    psar = talib.SAR(high, low)

    # Calculate CCI with default parameters
    cci = talib.CCI(high, low, close)

    # Determine signal for last OHLCV row
    if psar[-1] < close[-1] and cci[-1] > 100:
        return 1
    elif psar[-1] > close[-1] and cci[-1] < -100:
        return -1
    else:
        return 0

def psar_roc_strategy(data):
    # Define indicator parameters
    psar_af = 0.02
    psar_max_af = 0.2
    roc_period = 14
    
    # Calculate PSAR
    psar = tr.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'],step=psar_af, max_step=psar_max_af)
    psar_indicator = psar.psar()
    
    # Calculate ROC
    roc = ROCIndicator(close=data['Close'], window=roc_period)
    
    # Determine trading signals
    signals = []
    positions = []
    for i in range(1, len(data)):
        # Buy signal
        if data['Close'][i] > psar_indicator[i] and roc.roc()[i] > 0:
            signals.append(1)
        # Sell signal
        elif data['Close'][i] < psar_indicator[i] and roc.roc()[i] < 0:
            signals.append(-1)
        # No signal
        else:
            signals.append(0)
            
        
    
    # Determine signal for last OHLCV row
    buy_signal = data['Close'].iloc[-1] > psar_indicator.iloc[-1] and roc.roc().iloc[-1] > 0
    sell_signal = data['Close'].iloc[-1] < psar_indicator.iloc[-1] and roc.roc().iloc[-1] < 0
    
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_wpr_strategy(data):
    psar = tr.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'], step=0.02, max_step=0.2)
    wpr = mo.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'], lbp=14)
    
    buy_signal = psar.psar()[-1] > data['Close'][-1] and wpr.williams_r()[-1] < -80
    sell_signal = psar.psar()[-1] < data['Close'][-1] and wpr.williams_r()[-1] > -20
    
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_macd_hist_strategy(data):
    psar = tr.PSARIndicator(data['High'], data['Low'], data['Close'])
    macd = tr.MACD(data['Close'], window_fast=12, window_slow=26, window_signal=9)
    data['psar_up'] = psar.psar_up()
    data['psar_down'] = psar.psar_down()
    buy_signal = (data['psar_up'] < data['Close']) & (macd.macd_diff() > 0) & (macd.macd_signal() > 0)
    sell_signal = (data['psar_down'] > data['Close']) & (macd.macd_diff() < 0) & (macd.macd_signal() < 0)
    if buy_signal.any():
        return 1
    elif sell_signal.any():
        return -1
    else:
        return 0


def psar_macd_hist_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    # Calculate MACD
    macd, signal, hist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Determine buy/sell signals
    buy_signal = psar[-1] < data['Close'].iloc[-1] and hist[-1] > 0 and macd[-1] > signal[-1]
    sell_signal = psar[-1] > data['Close'].iloc[-1] and hist[-1] < 0 and macd[-1] < signal[-1]

    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_bbands_strategy(data):
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20)
    buy_signal = psar.iloc[-2] > data['Close'].iloc[-2] and psar.iloc[-1] < data['Close'].iloc[-1] and data['Close'].iloc[-1] > lower.iloc[-1]
    sell_signal = psar.iloc[-2] < data['Close'].iloc[-2] and psar.iloc[-1] > data['Close'].iloc[-1] and data['Close'].iloc[-1] < upper.iloc[-1]
    
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_atr_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate ATR
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Determine long and short positions
    positions = []
    for i in range(len(data)):
        if psar[i] < data['Close'][i] and data['Close'][i] - psar[i] > atr[i]:
            positions.append(1)
        elif psar[i] > data['Close'][i] and psar[i] - data['Close'][i] > atr[i]:
            positions.append(-1)
        else:
            positions.append(0)
    
    # Determine signal for last OHLCV row
    last_position = positions[-1]
    if last_position == 1:
        return 1
    elif last_position == -1:
        return -1
    else:
        return 0


def psar_stdev_strategy(data):
    # Calculate PSAR and STDEV
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    stdev = talib.STDDEV(data['Close'], timeperiod=20, nbdev=2)

    # Determine buy and sell signals
    buy_signal = (data['Close'] > psar) & (data['Close'] < psar + stdev)
    sell_signal = (data['Close'] < psar) & (data['Close'] > psar - stdev)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_kc_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate Keltner Channels
    keltner_mid = talib.SMA(data['Close'], timeperiod=20)
    keltner_upper = keltner_mid + 2*talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=10)
    keltner_lower = keltner_mid - 2*talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=10)

    # Determine signal for each row in data
    buy_signal = (psar > data['Close']) & (data['Low'] < keltner_lower)
    sell_signal = (psar < data['Close']) & (data['High'] > keltner_upper)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def psar_donchian_strategy(data):
    # Calculate Parabolic SAR indicator
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    # Calculate Donchian Channel indicator
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=20)

    # Determine buy/sell signals
    buy_signal = (psar > data['Close']) & (data['Close'] > upper)
    sell_signal = (psar < data['Close']) & (data['Close'] < lower)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_chandelier_exit_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate Chandelier Exit
    atr = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=22)
    long_chandelier_exit = data['High'].rolling(window=22).max() - atr * 3
    short_chandelier_exit = data['Low'].rolling(window=22).min() + atr * 3
    
    # Determine buy and sell signals
    buy_signal = (data['Close'] > psar) & (data['Low'] > long_chandelier_exit)
    sell_signal = (data['Close'] < psar) & (data['High'] < short_chandelier_exit)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def psar_obv_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    # Calculate OBV
    obv = talib.OBV(data['Close'], data['Volume'])

    # Determine buy and sell signals
    buy_signal = (psar < data['Close']) & (obv > obv.shift(1))
    sell_signal = (psar > data['Close']) & (obv < obv.shift(1))

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_cmf_strategy(data):
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    cmf = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).chaikin_money_flow()
    
    buy_signal = (psar > data['Close']) & (cmf > 0)
    sell_signal = (psar < data['Close']) & (cmf < 0)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_vroc_strategy(data):
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    vroc = talib.ROC(data['Volume'], timeperiod=14)
    
    buy_signal = psar < data['Close'] # PSAR below Close
    sell_signal = psar > data['Close'] # PSAR above Close
    
    if buy_signal.iloc[-1] and vroc.iloc[-1] > 0:
        return 1 # buy signal
    elif sell_signal.iloc[-1] and vroc.iloc[-1] < 0:
        return -1 # sell signal
    else:
        return 0 # no signal

def psar_mfi_strategy(data):
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    mfi = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)

    buy_signal = (psar > data['Close']) & (mfi < 20)
    sell_signal = (psar < data['Close']) & (mfi > 80)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def psar_adl_strategy(df):
    psar = tr.PSARIndicator(df['High'], df['Low'], df['Close'])
    adl = AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
    
    buy_signal = (df['Close'].iloc[-1] > psar.psar().iloc[-1]) & (adl.acc_dist_index().iloc[-1] > adl.acc_dist_index().iloc[-2])
    sell_signal = (df['Close'].iloc[-1] < psar.psar().iloc[-1]) & (adl.acc_dist_index().iloc[-1] < adl.acc_dist_index().iloc[-2])
    
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_eom_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate EOM
    eom = EaseOfMovementIndicator(
        high=data['High'], low=data['Low'], volume=data['Volume'], window=14, fillna=False
    ).sma_ease_of_movement()
    
    # Generate buy and sell signals
    buy_signal = (psar.iloc[-1] > data['Close'].iloc[-1]) & (eom.iloc[-1] > 0)
    sell_signal = (psar.iloc[-1] < data['Close'].iloc[-1]) & (eom.iloc[-1] < 0)
    
    # Determine signal for last OHLCV row
    if buy_signal:
        return 1
    elif sell_signal:
        return -1
    else:
        return 0


def psar_pivot_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate Pivot Points
    pp = (data['High'] + data['Low'] + data['Close']) / 3
    r1 = 2 * pp - data['Low']
    s1 = 2 * pp - data['High']
    r2 = pp + (data['High'] - data['Low'])
    s2 = pp - (data['High'] - data['Low'])
    r3 = data['High'] + 2 * (pp - data['Low'])
    s3 = data['Low'] - 2 * (data['High'] - pp)
    
    # Determine trading signals
    buy_signal = (psar > data['Close']) & (data['Close'] > s1)
    sell_signal = (psar < data['Close']) & (data['Close'] < r1)
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0

def psar_fibonacci_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    # Calculate Fibonacci levels
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    fib_23_6 = low + diff * 0.236
    fib_38_2 = low + diff * 0.382
    fib_50_0 = low + diff * 0.5
    fib_61_8 = low + diff * 0.618
    fib_76_4 = low + diff * 0.764

    # Determine buy and sell signals
    buy_signal = (data['Close'] > psar) & (data['Close'] > fib_61_8)
    sell_signal = (data['Close'] < psar) & (data['Close'] < fib_38_2)

    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


def psar_srl_strategy(data):
    # Calculate PSAR
	high = data['High']
	low = data['Low']
	close = data['Close']
	psar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # Calculate SRL
	resistance = talib.MAX(high, timeperiod=14)
	support = talib.MIN(low, timeperiod=14)

	# Determine signal for last OHLCV row
	if close.iloc[-1] > psar[-1] and close.iloc[-1] > resistance.iloc[-1]:
		return 1
	elif close.iloc[-1] < psar[-1] and close.iloc[-1] < support.iloc[-1]:
		return -1
	else:
		return 0


def psar_gann_strategy(data):
    # Calculate PSAR
    psar = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate Gann Lines
    high_max = data['High'].rolling(window=20).max()
    low_min = data['Low'].rolling(window=20).min()
    diff = high_max - low_min
    levels = [low_min[i] + diff[i] * j / 8 for i in range(20) for j in range(1, 8)]
    
    # Determine buy and sell signals
    buy_signal = (data['Close'] > psar) & (data['Close'] > levels[-1])
    sell_signal = (data['Close'] < psar) & (data['Close'] < levels[0])
    
    # Determine signal for last OHLCV row
    if buy_signal.iloc[-1]:
        return 1
    elif sell_signal.iloc[-1]:
        return -1
    else:
        return 0


# TODO UPDATE TO THESE - old list has duplicates
# trend_indicators = ['SMA', 'EMA', 'MACD', 'PSAR', 'ICHIMOKU', 'SUPERTREND']
# momentum_indicators = ['RSI', 'ROC', 'CMO', 'PPO', 'WPR', 'RVI']
# volatility_indicators = ['BBANDS', 'ATR', 'STDEV', 'KC', 'Donchian', 'Chandelier_Exit']
# volume_indicators = ['OBV', 'CMF', 'VROC', 'MFI', 'ADL', 'EOM']
# support_resistance_indicators = ['Pivot_Points', 'Fibonacci_Retracement', 'SRL', 'Gann_Lines', 'Andrews_Pitchfork', 'MA_Support_Resistance']
# oscillator_indicators = ['VWAP', 'TSI', 'Stochastic', 'Awesome_Oscillator', 'DPO', 'CCI']

# Define the labels for the heatmap
trend_indicators = ['SMA', 'EMA', 'MACD', 'PSAR', 'ICHIMOKU', 'SUPERTREND']
momentum_indicators = ['RSI', 'Stochastic', 'CCI', 'ROC', 'WPR', 'MACD_Hist']
volatility_indicators = ['BBANDS', 'ATR', 'STDEV', 'KC', 'Donchian', 'Chandelier_Exit']
volume_indicators = ['OBV', 'CMF', 'VROC', 'MFI', 'ADL', 'EOM']
support_resistance_indicators = ['Pivot_Points', 'Fibonacci_Retracement', 'SRL', 'Gann_Lines', 'Andrews_Pitchfork', 'MA_Support_Resistance']
oscillator_indicators = ['RSI', 'MACD', 'Stochastic', 'Awesome_Oscillator', 'WPR', 'CCI']

# Define the labels for the indicators
indicators = trend_indicators + momentum_indicators + volatility_indicators + volume_indicators + support_resistance_indicators + oscillator_indicators
data = pd.read_csv("EURUSD=X.csv", index_col=0)
# Define the indicator groups
indicators_groups = {
    'Trend': trend_indicators,
    'Momentum': momentum_indicators,
    'Volatility': volatility_indicators,
    'Volume': volume_indicators,
    'S & R': support_resistance_indicators,
    'Oscillator': oscillator_indicators
}
# Define the stock symbol and time period to download

# symbol = "EURUSD=X"
# end_date = datetime.now().strftime('%Y-%m-%d')
# start_date = (datetime.now() - timedelta(days=1440)).strftime('%Y-%m-%d')

# # Download the historical data from Yahoo Finance
# data = yf.download(symbol, start=start_date, end=end_date)

# data.to_csv(f"{symbol}.csv") 


# Define a dictionary of strategies for each indicator combination
strategies = {
	('*', '*'): perceptron_strategy(data),
	('SMA', 'SMA'): sma_sma_strategy(data),
	('EMA', 'EMA'): ema_ema_strategy(data),
	('SMA', 'EMA'): sma_ema_strategy(data),
	('SMA', 'MACD'): sma_macd_strategy(data),
	('SMA', 'PSAR'): sma_psar_strategy(data),
	('SMA', 'ICHIMOKU'): sma_ichimoku_strategy(data),
	('SMA', 'SUPERTREND'): sma_supertrend_strategy(data),
	('SMA', 'RSI'): sma_rsi_strategy(data),
	('SMA', 'Stochastic'): sma_stochastic_strategy(data),
	('SMA', 'CCI'): sma_cci_strategy(data),
	('SMA', 'ROC'): sma_roc_strategy(data),
	('SMA', 'WPR'): sma_wpr_strategy(data),
	('SMA', 'MACD_Hist'): sma_macd_hist_strategy(data),
	('SMA', 'BBANDS'): sma_bbands_strategy(data),
	('SMA', 'ATR'): sma_atr_strategy(data),
	('SMA', 'STDEV'): sma_stdev_strategy(data),
	('SMA', 'KC'): sma_kc_strategy(data),
	('SMA', 'Donchian'): sma_donchian_strategy(data),
	('SMA', 'Chandelier_Exit'): sma_chandelier_exit_strategy(data),
	('SMA', 'OBV'): sma_obv_strategy(data),
	('SMA', 'CMF'): sma_cmf_strategy(data),
	('SMA', 'VROC'): sma_vroc_strategy(data),
	('SMA', 'MFI'): sma_mfi_strategy(data),
	('SMA', 'ADL'): sma_adl_strategy(data),
	('SMA', 'EOM'): sma_eom_strategy(data),
	('SMA', 'Pivot_Points'): sma_pivot_points_strategy(data),
	('SMA', 'Fibonacci_Retracement'): sma_fibonacci_retracement_strategy(data),
	('SMA', 'SRL'): sma_srl_strategy(data),
	('SMA', 'Gann_Lines'): sma_gann_lines_strategy(data),
	('SMA', 'Andrews_Pitchfork'): sma_andrews_pitchfork_strategy(data),
	('SMA', 'MA_Support_Resistance'): sma_ma_support_resistance_strategy(data),
	('SMA', 'Awesome_Oscillator'): sma_awesome_oscillator_strategy(data),
	('EMA', 'SMA'): sma_ema_strategy(data),
	('EMA', 'MACD'): ema_macd_strategy(data),
	('EMA', 'PSAR'): ema_psar_strategy(data),
	('EMA', 'ICHIMOKU'): ema_ichimoku_strategy(data),
	('EMA', 'SUPERTREND'): ema_supertrend_strategy(data),
	('EMA', 'RSI'): ema_rsi_strategy(data),
	('EMA', 'Stochastic'): ema_stochastic_strategy(data),
	('EMA', 'CCI'): ema_cci_strategy(data),
	('EMA', 'ROC'): ema_roc_strategy(data),
	('EMA', 'WPR'): ema_wpr_strategy(data),
	('EMA', 'MACD_Hist'): ema_macd_hist_strategy(data),
	('EMA', 'BBANDS'): ema_bbands_strategy(data),
	('EMA', 'ATR'): ema_atr_strategy(data),
	('EMA', 'STDEV'): ema_stdev_strategy(data),
	('EMA', 'KC'): ema_kc_strategy(data),
	('EMA', 'Donchian'): ema_donchian_strategy(data),
	('EMA', 'Chandelier_Exit'): ema_chandelier_exit_strategy(data),
	('EMA', 'OBV'): ema_obv_strategy(data),
	('EMA', 'CMF'): ema_cmf_strategy(data),
	('EMA', 'VROC'): ema_vroc_strategy(data),
	('EMA', 'MFI'): ema_mfi_strategy(data),
	('EMA', 'ADL'): ema_adl_strategy(data),
	('EMA', 'EOM'): ema_eom_strategy(data),
	('EMA', 'Pivot_Points'): ema_pivot_points_strategy(data),
	('EMA', 'Fibonacci_Retracement'): ema_fibonacci_strategy(data),
	('EMA', 'SRL'): ema_srl_strategy(data),
	('EMA', 'Gann_Lines'): ema_gann_lines_strategy(data),
	('EMA', 'Andrews_Pitchfork'): ema_andrews_pitchfork_strategy(data),
	('EMA', 'MA_Support_Resistance'): ema_sr_strategy(data),
	('EMA', 'Awesome_Oscillator'): ema_awesome_oscillator_strategy(data),
	('MACD', 'SMA'): sma_macd_strategy(data),
	('MACD', 'EMA'): ema_macd_strategy(data),
	('MACD', 'MACD'): macd_strategy(data),
	('MACD', 'PSAR'): macd_psar_strategy(data),
	('MACD', 'ICHIMOKU'): macd_ichimoku_strategy(data),
	('MACD', 'SUPERTREND'): macd_supertrend_atr_strategy(data),
	('MACD', 'RSI'): macd_rsi_strategy(data),
	('MACD', 'Stochastic'): macd_stochastic_strategy(data),
	('MACD', 'CCI'): macd_cci_strategy(data),
	('MACD', 'ROC'): macd_roc_strategy(data),
	('MACD', 'WPR'): macd_wpr_strategy(data),
	('MACD', 'MACD_Hist'): macd_hist_strategy(data),
	('MACD', 'BBANDS'): macd_bbands_strategy(data),
	('MACD', 'ATR'): macd_atr_strategy(data),
	('MACD', 'STDEV'): macd_stdev_strategy(data),
	('MACD', 'KC'): macd_kc_strategy(data),
	('MACD', 'Donchian'): macd_donchian_strategy(data),
	('MACD', 'Chandelier_Exit'): macd_chandelier_exit_strategy(data),
	('MACD', 'OBV'): macd_obv_strategy(data),
	('MACD', 'CMF'): macd_cmf_strategy(data),
	('MACD', 'VROC'): macd_vroc_strategy(data),
	('MACD', 'MFI'): macd_mfi_strategy(data),
	('MACD', 'ADL'): macd_adl_strategy(data),
	('MACD', 'EOM'): macd_eom_strategy(data),
	('MACD', 'Pivot_Points'): macd_pivot_points_strategy(data),
	('MACD', 'Fibonacci_Retracement'): macd_fibonacci_strategy(data),
	('MACD', 'SRL'): macd_srl_strategy(data),
	('MACD', 'Gann_Lines'): macd_gann_lines_strategy(data),
	('MACD', 'Andrews_Pitchfork'): macd_andrews_pitchfork_strategy(data),
	('MACD', 'MA_Support_Resistance'): macd_ma_sr_strategy(data),
	('MACD', 'Awesome_Oscillator'): macd_awesome_oscillator_strategy(data),
	('PSAR', 'SMA'): sma_psar_strategy(data),
	('PSAR', 'EMA'): ema_psar_strategy(data),
	('PSAR', 'MACD'): macd_psar_strategy(data),
	('PSAR', 'PSAR'): psar_strategy(data),
	('PSAR', 'ICHIMOKU'): psar_ichimoku_strategy(data),
	('PSAR', 'SUPERTREND'): psar_supertrend_strategy(data),
	('PSAR', 'RSI'): psar_rsi_strategy(data),
	('PSAR', 'Stochastic'): psar_stochastic_strategy(data),
	('PSAR', 'CCI'): psar_cci_strategy(data),
	('PSAR', 'ROC'): psar_roc_strategy(data),
	('PSAR', 'WPR'): psar_wpr_strategy(data),
	('PSAR', 'MACD_Hist'): psar_macd_hist_strategy(data),
	('PSAR', 'BBANDS'): psar_bbands_strategy(data),
	('PSAR', 'ATR'): psar_atr_strategy(data),
	('PSAR', 'STDEV'): psar_stdev_strategy(data),
	('PSAR', 'KC'): psar_kc_strategy(data),
	('PSAR', 'Donchian'): psar_donchian_strategy(data),
	('PSAR', 'Chandelier_Exit'): psar_chandelier_exit_strategy(data),
	('PSAR', 'OBV'): psar_obv_strategy(data),
	('PSAR', 'CMF'): psar_cmf_strategy(data),
	('PSAR', 'VROC'): psar_vroc_strategy(data),
	('PSAR', 'MFI'): psar_mfi_strategy(data),
	('PSAR', 'ADL'): psar_adl_strategy(data),
	('PSAR', 'EOM'): psar_eom_strategy(data),
	('PSAR', 'Pivot_Points'): psar_pivot_strategy(data),
	('PSAR', 'Fibonacci_Retracement'): psar_fibonacci_strategy(data),
	('PSAR', 'SRL'): psar_srl_strategy(data),
	('PSAR', 'Gann_Lines'): psar_gann_strategy(data),
	('PSAR', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('PSAR', 'MA_Support_Resistance'): perceptron_strategy(data),
	('PSAR', 'Awesome_Oscillator'): perceptron_strategy(data),
	('ICHIMOKU', 'SMA'): sma_ichimoku_strategy(data),
	('ICHIMOKU', 'EMA'): ema_ichimoku_strategy(data),
	('ICHIMOKU', 'MACD'): macd_ichimoku_strategy(data),
	('ICHIMOKU', 'PSAR'): psar_ichimoku_strategy(data),
	('ICHIMOKU', 'ICHIMOKU'): perceptron_strategy(data),
	('ICHIMOKU', 'SUPERTREND'): perceptron_strategy(data),
	('ICHIMOKU', 'RSI'): perceptron_strategy(data),
	('ICHIMOKU', 'Stochastic'): perceptron_strategy(data),
	('ICHIMOKU', 'CCI'): perceptron_strategy(data),
	('ICHIMOKU', 'ROC'): perceptron_strategy(data),
	('ICHIMOKU', 'WPR'): perceptron_strategy(data),
	('ICHIMOKU', 'MACD_Hist'): perceptron_strategy(data),
	('ICHIMOKU', 'BBANDS'): perceptron_strategy(data),
	('ICHIMOKU', 'ATR'): perceptron_strategy(data),
	('ICHIMOKU', 'STDEV'): perceptron_strategy(data),
	('ICHIMOKU', 'KC'): perceptron_strategy(data),
	('ICHIMOKU', 'Donchian'): perceptron_strategy(data),
	('ICHIMOKU', 'Chandelier_Exit'): perceptron_strategy(data),
	('ICHIMOKU', 'OBV'): perceptron_strategy(data),
	('ICHIMOKU', 'CMF'): perceptron_strategy(data),
	('ICHIMOKU', 'VROC'): perceptron_strategy(data),
	('ICHIMOKU', 'MFI'): perceptron_strategy(data),
	('ICHIMOKU', 'ADL'): perceptron_strategy(data),
	('ICHIMOKU', 'EOM'): perceptron_strategy(data),
	('ICHIMOKU', 'Pivot_Points'): perceptron_strategy(data),
	('ICHIMOKU', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('ICHIMOKU', 'SRL'): perceptron_strategy(data),
	('ICHIMOKU', 'Gann_Lines'): perceptron_strategy(data),
	('ICHIMOKU', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('ICHIMOKU', 'MA_Support_Resistance'): perceptron_strategy(data),
	('ICHIMOKU', 'Awesome_Oscillator'): perceptron_strategy(data),
	('SUPERTREND', 'SMA'): sma_supertrend_strategy(data),
	('SUPERTREND', 'EMA'): ema_supertrend_strategy(data),
	('SUPERTREND', 'MACD'): macd_supertrend_atr_strategy(data),
	('SUPERTREND', 'PSAR'): psar_supertrend_strategy(data),
	('SUPERTREND', 'ICHIMOKU'): perceptron_strategy(data),
	('SUPERTREND', 'SUPERTREND'): perceptron_strategy(data),
	('SUPERTREND', 'RSI'): perceptron_strategy(data),
	('SUPERTREND', 'Stochastic'): perceptron_strategy(data),
	('SUPERTREND', 'CCI'): perceptron_strategy(data),
	('SUPERTREND', 'ROC'): perceptron_strategy(data),
	('SUPERTREND', 'WPR'): perceptron_strategy(data),
	('SUPERTREND', 'MACD_Hist'): perceptron_strategy(data),
	('SUPERTREND', 'BBANDS'): perceptron_strategy(data),
	('SUPERTREND', 'ATR'): perceptron_strategy(data),
	('SUPERTREND', 'STDEV'): perceptron_strategy(data),
	('SUPERTREND', 'KC'): perceptron_strategy(data),
	('SUPERTREND', 'Donchian'): perceptron_strategy(data),
	('SUPERTREND', 'Chandelier_Exit'): perceptron_strategy(data),
	('SUPERTREND', 'OBV'): perceptron_strategy(data),
	('SUPERTREND', 'CMF'): perceptron_strategy(data),
	('SUPERTREND', 'VROC'): perceptron_strategy(data),
	('SUPERTREND', 'MFI'): perceptron_strategy(data),
	('SUPERTREND', 'ADL'): perceptron_strategy(data),
	('SUPERTREND', 'EOM'): perceptron_strategy(data),
	('SUPERTREND', 'Pivot_Points'): perceptron_strategy(data),
	('SUPERTREND', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('SUPERTREND', 'SRL'): perceptron_strategy(data),
	('SUPERTREND', 'Gann_Lines'): perceptron_strategy(data),
	('SUPERTREND', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('SUPERTREND', 'MA_Support_Resistance'): perceptron_strategy(data),
	('SUPERTREND', 'Awesome_Oscillator'): perceptron_strategy(data),
	('RSI', 'SMA'): sma_rsi_strategy(data),
	('RSI', 'EMA'): ema_rsi_strategy(data),
	('RSI', 'MACD'): macd_rsi_strategy(data),
	('RSI', 'PSAR'): psar_rsi_strategy(data),
	('RSI', 'ICHIMOKU'): perceptron_strategy(data),
	('RSI', 'SUPERTREND'): perceptron_strategy(data),
	('RSI', 'RSI'): perceptron_strategy(data),
	('RSI', 'Stochastic'): perceptron_strategy(data),
	('RSI', 'CCI'): perceptron_strategy(data),
	('RSI', 'ROC'): perceptron_strategy(data),
	('RSI', 'WPR'): perceptron_strategy(data),
	('RSI', 'MACD_Hist'): perceptron_strategy(data),
	('RSI', 'BBANDS'): perceptron_strategy(data),
	('RSI', 'ATR'): perceptron_strategy(data),
	('RSI', 'STDEV'): perceptron_strategy(data),
	('RSI', 'KC'): perceptron_strategy(data),
	('RSI', 'Donchian'): perceptron_strategy(data),
	('RSI', 'Chandelier_Exit'): perceptron_strategy(data),
	('RSI', 'OBV'): perceptron_strategy(data),
	('RSI', 'CMF'): perceptron_strategy(data),
	('RSI', 'VROC'): perceptron_strategy(data),
	('RSI', 'MFI'): perceptron_strategy(data),
	('RSI', 'ADL'): perceptron_strategy(data),
	('RSI', 'EOM'): perceptron_strategy(data),
	('RSI', 'Pivot_Points'): perceptron_strategy(data),
	('RSI', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('RSI', 'SRL'): perceptron_strategy(data),
	('RSI', 'Gann_Lines'): perceptron_strategy(data),
	('RSI', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('RSI', 'MA_Support_Resistance'): perceptron_strategy(data),
	('RSI', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Stochastic', 'SMA'): sma_stochastic_strategy(data),
	('Stochastic', 'EMA'): ema_stochastic_strategy(data),
	('Stochastic', 'MACD'): macd_stochastic_strategy(data),
	('Stochastic', 'PSAR'): psar_stochastic_strategy(data),
	('Stochastic', 'ICHIMOKU'): perceptron_strategy(data),
	('Stochastic', 'SUPERTREND'): perceptron_strategy(data),
	('Stochastic', 'RSI'): perceptron_strategy(data),
	('Stochastic', 'Stochastic'): perceptron_strategy(data),
	('Stochastic', 'CCI'): perceptron_strategy(data),
	('Stochastic', 'ROC'): perceptron_strategy(data),
	('Stochastic', 'WPR'): perceptron_strategy(data),
	('Stochastic', 'MACD_Hist'): perceptron_strategy(data),
	('Stochastic', 'BBANDS'): perceptron_strategy(data),
	('Stochastic', 'ATR'): perceptron_strategy(data),
	('Stochastic', 'STDEV'): perceptron_strategy(data),
	('Stochastic', 'KC'): perceptron_strategy(data),
	('Stochastic', 'Donchian'): perceptron_strategy(data),
	('Stochastic', 'Chandelier_Exit'): perceptron_strategy(data),
	('Stochastic', 'OBV'): perceptron_strategy(data),
	('Stochastic', 'CMF'): perceptron_strategy(data),
	('Stochastic', 'VROC'): perceptron_strategy(data),
	('Stochastic', 'MFI'): perceptron_strategy(data),
	('Stochastic', 'ADL'): perceptron_strategy(data),
	('Stochastic', 'EOM'): perceptron_strategy(data),
	('Stochastic', 'Pivot_Points'): perceptron_strategy(data),
	('Stochastic', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Stochastic', 'SRL'): perceptron_strategy(data),
	('Stochastic', 'Gann_Lines'): perceptron_strategy(data),
	('Stochastic', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Stochastic', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Stochastic', 'Awesome_Oscillator'): perceptron_strategy(data),
	('CCI', 'SMA'): sma_cci_strategy(data),
	('CCI', 'EMA'): ema_cci_strategy(data),
	('CCI', 'MACD'): macd_cci_strategy(data),
	('CCI', 'PSAR'): psar_cci_strategy(data),
	('CCI', 'ICHIMOKU'): perceptron_strategy(data),
	('CCI', 'SUPERTREND'): perceptron_strategy(data),
	('CCI', 'RSI'): perceptron_strategy(data),
	('CCI', 'Stochastic'): perceptron_strategy(data),
	('CCI', 'CCI'): perceptron_strategy(data),
	('CCI', 'ROC'): perceptron_strategy(data),
	('CCI', 'WPR'): perceptron_strategy(data),
	('CCI', 'MACD_Hist'): perceptron_strategy(data),
	('CCI', 'BBANDS'): perceptron_strategy(data),
	('CCI', 'ATR'): perceptron_strategy(data),
	('CCI', 'STDEV'): perceptron_strategy(data),
	('CCI', 'KC'): perceptron_strategy(data),
	('CCI', 'Donchian'): perceptron_strategy(data),
	('CCI', 'Chandelier_Exit'): perceptron_strategy(data),
	('CCI', 'OBV'): perceptron_strategy(data),
	('CCI', 'CMF'): perceptron_strategy(data),
	('CCI', 'VROC'): perceptron_strategy(data),
	('CCI', 'MFI'): perceptron_strategy(data),
	('CCI', 'ADL'): perceptron_strategy(data),
	('CCI', 'EOM'): perceptron_strategy(data),
	('CCI', 'Pivot_Points'): perceptron_strategy(data),
	('CCI', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('CCI', 'SRL'): perceptron_strategy(data),
	('CCI', 'Gann_Lines'): perceptron_strategy(data),
	('CCI', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('CCI', 'MA_Support_Resistance'): perceptron_strategy(data),
	('CCI', 'Awesome_Oscillator'): perceptron_strategy(data),
	('ROC', 'SMA'): sma_roc_strategy(data),
	('ROC', 'EMA'): ema_roc_strategy(data),
	('ROC', 'MACD'): macd_roc_strategy(data),
	('ROC', 'PSAR'): psar_roc_strategy(data),
	('ROC', 'ICHIMOKU'): perceptron_strategy(data),
	('ROC', 'SUPERTREND'): perceptron_strategy(data),
	('ROC', 'RSI'): perceptron_strategy(data),
	('ROC', 'Stochastic'): perceptron_strategy(data),
	('ROC', 'CCI'): perceptron_strategy(data),
	('ROC', 'ROC'): perceptron_strategy(data),
	('ROC', 'WPR'): perceptron_strategy(data),
	('ROC', 'MACD_Hist'): perceptron_strategy(data),
	('ROC', 'BBANDS'): perceptron_strategy(data),
	('ROC', 'ATR'): perceptron_strategy(data),
	('ROC', 'STDEV'): perceptron_strategy(data),
	('ROC', 'KC'): perceptron_strategy(data),
	('ROC', 'Donchian'): perceptron_strategy(data),
	('ROC', 'Chandelier_Exit'): perceptron_strategy(data),
	('ROC', 'OBV'): perceptron_strategy(data),
	('ROC', 'CMF'): perceptron_strategy(data),
	('ROC', 'VROC'): perceptron_strategy(data),
	('ROC', 'MFI'): perceptron_strategy(data),
	('ROC', 'ADL'): perceptron_strategy(data),
	('ROC', 'EOM'): perceptron_strategy(data),
	('ROC', 'Pivot_Points'): perceptron_strategy(data),
	('ROC', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('ROC', 'SRL'): perceptron_strategy(data),
	('ROC', 'Gann_Lines'): perceptron_strategy(data),
	('ROC', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('ROC', 'MA_Support_Resistance'): perceptron_strategy(data),
	('ROC', 'Awesome_Oscillator'): perceptron_strategy(data),
	('WPR', 'SMA'): sma_wpr_strategy(data),
	('WPR', 'EMA'): ema_wpr_strategy(data),
	('WPR', 'MACD'): macd_wpr_strategy(data),
	('WPR', 'PSAR'): psar_wpr_strategy(data),
	('WPR', 'ICHIMOKU'): perceptron_strategy(data),
	('WPR', 'SUPERTREND'): perceptron_strategy(data),
	('WPR', 'RSI'): perceptron_strategy(data),
	('WPR', 'Stochastic'): perceptron_strategy(data),
	('WPR', 'CCI'): perceptron_strategy(data),
	('WPR', 'ROC'): perceptron_strategy(data),
	('WPR', 'WPR'): perceptron_strategy(data),
	('WPR', 'MACD_Hist'): perceptron_strategy(data),
	('WPR', 'BBANDS'): perceptron_strategy(data),
	('WPR', 'ATR'): perceptron_strategy(data),
	('WPR', 'STDEV'): perceptron_strategy(data),
	('WPR', 'KC'): perceptron_strategy(data),
	('WPR', 'Donchian'): perceptron_strategy(data),
	('WPR', 'Chandelier_Exit'): perceptron_strategy(data),
	('WPR', 'OBV'): perceptron_strategy(data),
	('WPR', 'CMF'): perceptron_strategy(data),
	('WPR', 'VROC'): perceptron_strategy(data),
	('WPR', 'MFI'): perceptron_strategy(data),
	('WPR', 'ADL'): perceptron_strategy(data),
	('WPR', 'EOM'): perceptron_strategy(data),
	('WPR', 'Pivot_Points'): perceptron_strategy(data),
	('WPR', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('WPR', 'SRL'): perceptron_strategy(data),
	('WPR', 'Gann_Lines'): perceptron_strategy(data),
	('WPR', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('WPR', 'MA_Support_Resistance'): perceptron_strategy(data),
	('WPR', 'Awesome_Oscillator'): perceptron_strategy(data),
	('MACD_Hist', 'SMA'): sma_macd_hist_strategy(data),
	('MACD_Hist', 'EMA'): ema_macd_hist_strategy(data),
	('MACD_Hist', 'MACD'): macd_hist_strategy(data),
	('MACD_Hist', 'PSAR'): psar_macd_hist_strategy(data),
	('MACD_Hist', 'ICHIMOKU'): perceptron_strategy(data),
	('MACD_Hist', 'SUPERTREND'): perceptron_strategy(data),
	('MACD_Hist', 'RSI'): perceptron_strategy(data),
	('MACD_Hist', 'Stochastic'): perceptron_strategy(data),
	('MACD_Hist', 'CCI'): perceptron_strategy(data),
	('MACD_Hist', 'ROC'): perceptron_strategy(data),
	('MACD_Hist', 'WPR'): perceptron_strategy(data),
	('MACD_Hist', 'MACD_Hist'): perceptron_strategy(data),
	('MACD_Hist', 'BBANDS'): perceptron_strategy(data),
	('MACD_Hist', 'ATR'): perceptron_strategy(data),
	('MACD_Hist', 'STDEV'): perceptron_strategy(data),
	('MACD_Hist', 'KC'): perceptron_strategy(data),
	('MACD_Hist', 'Donchian'): perceptron_strategy(data),
	('MACD_Hist', 'Chandelier_Exit'): perceptron_strategy(data),
	('MACD_Hist', 'OBV'): perceptron_strategy(data),
	('MACD_Hist', 'CMF'): perceptron_strategy(data),
	('MACD_Hist', 'VROC'): perceptron_strategy(data),
	('MACD_Hist', 'MFI'): perceptron_strategy(data),
	('MACD_Hist', 'ADL'): perceptron_strategy(data),
	('MACD_Hist', 'EOM'): perceptron_strategy(data),
	('MACD_Hist', 'Pivot_Points'): perceptron_strategy(data),
	('MACD_Hist', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('MACD_Hist', 'SRL'): perceptron_strategy(data),
	('MACD_Hist', 'Gann_Lines'): perceptron_strategy(data),
	('MACD_Hist', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('MACD_Hist', 'MA_Support_Resistance'): perceptron_strategy(data),
	('MACD_Hist', 'Awesome_Oscillator'): perceptron_strategy(data),
	('BBANDS', 'SMA'): sma_bbands_strategy(data),
	('BBANDS', 'EMA'): ema_bbands_strategy(data),
	('BBANDS', 'MACD'): macd_bbands_strategy(data),
	('BBANDS', 'PSAR'): psar_bbands_strategy(data),
	('BBANDS', 'ICHIMOKU'): perceptron_strategy(data),
	('BBANDS', 'SUPERTREND'): perceptron_strategy(data),
	('BBANDS', 'RSI'): perceptron_strategy(data),
	('BBANDS', 'Stochastic'): perceptron_strategy(data),
	('BBANDS', 'CCI'): perceptron_strategy(data),
	('BBANDS', 'ROC'): perceptron_strategy(data),
	('BBANDS', 'WPR'): perceptron_strategy(data),
	('BBANDS', 'MACD_Hist'): perceptron_strategy(data),
	('BBANDS', 'BBANDS'): perceptron_strategy(data),
	('BBANDS', 'ATR'): perceptron_strategy(data),
	('BBANDS', 'STDEV'): perceptron_strategy(data),
	('BBANDS', 'KC'): perceptron_strategy(data),
	('BBANDS', 'Donchian'): perceptron_strategy(data),
	('BBANDS', 'Chandelier_Exit'): perceptron_strategy(data),
	('BBANDS', 'OBV'): perceptron_strategy(data),
	('BBANDS', 'CMF'): perceptron_strategy(data),
	('BBANDS', 'VROC'): perceptron_strategy(data),
	('BBANDS', 'MFI'): perceptron_strategy(data),
	('BBANDS', 'ADL'): perceptron_strategy(data),
	('BBANDS', 'EOM'): perceptron_strategy(data),
	('BBANDS', 'Pivot_Points'): perceptron_strategy(data),
	('BBANDS', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('BBANDS', 'SRL'): perceptron_strategy(data),
	('BBANDS', 'Gann_Lines'): perceptron_strategy(data),
	('BBANDS', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('BBANDS', 'MA_Support_Resistance'): perceptron_strategy(data),
	('BBANDS', 'Awesome_Oscillator'): perceptron_strategy(data),
	('ATR', 'SMA'): sma_atr_strategy(data),
	('ATR', 'EMA'): ema_atr_strategy(data),
	('ATR', 'MACD'): macd_atr_strategy(data),
	('ATR', 'PSAR'): psar_atr_strategy(data),
	('ATR', 'ICHIMOKU'): perceptron_strategy(data),
	('ATR', 'SUPERTREND'): perceptron_strategy(data),
	('ATR', 'RSI'): perceptron_strategy(data),
	('ATR', 'Stochastic'): perceptron_strategy(data),
	('ATR', 'CCI'): perceptron_strategy(data),
	('ATR', 'ROC'): perceptron_strategy(data),
	('ATR', 'WPR'): perceptron_strategy(data),
	('ATR', 'MACD_Hist'): perceptron_strategy(data),
	('ATR', 'BBANDS'): perceptron_strategy(data),
	('ATR', 'ATR'): perceptron_strategy(data),
	('ATR', 'STDEV'): perceptron_strategy(data),
	('ATR', 'KC'): perceptron_strategy(data),
	('ATR', 'Donchian'): perceptron_strategy(data),
	('ATR', 'Chandelier_Exit'): perceptron_strategy(data),
	('ATR', 'OBV'): perceptron_strategy(data),
	('ATR', 'CMF'): perceptron_strategy(data),
	('ATR', 'VROC'): perceptron_strategy(data),
	('ATR', 'MFI'): perceptron_strategy(data),
	('ATR', 'ADL'): perceptron_strategy(data),
	('ATR', 'EOM'): perceptron_strategy(data),
	('ATR', 'Pivot_Points'): perceptron_strategy(data),
	('ATR', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('ATR', 'SRL'): perceptron_strategy(data),
	('ATR', 'Gann_Lines'): perceptron_strategy(data),
	('ATR', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('ATR', 'MA_Support_Resistance'): perceptron_strategy(data),
	('ATR', 'Awesome_Oscillator'): perceptron_strategy(data),
	('STDEV', 'SMA'): sma_stdev_strategy(data),
	('STDEV', 'EMA'): ema_stdev_strategy(data),
	('STDEV', 'MACD'): macd_stdev_strategy(data),
	('STDEV', 'PSAR'): psar_stdev_strategy(data),
	('STDEV', 'ICHIMOKU'): perceptron_strategy(data),
	('STDEV', 'SUPERTREND'): perceptron_strategy(data),
	('STDEV', 'RSI'): perceptron_strategy(data),
	('STDEV', 'Stochastic'): perceptron_strategy(data),
	('STDEV', 'CCI'): perceptron_strategy(data),
	('STDEV', 'ROC'): perceptron_strategy(data),
	('STDEV', 'WPR'): perceptron_strategy(data),
	('STDEV', 'MACD_Hist'): perceptron_strategy(data),
	('STDEV', 'BBANDS'): perceptron_strategy(data),
	('STDEV', 'ATR'): perceptron_strategy(data),
	('STDEV', 'STDEV'): perceptron_strategy(data),
	('STDEV', 'KC'): perceptron_strategy(data),
	('STDEV', 'Donchian'): perceptron_strategy(data),
	('STDEV', 'Chandelier_Exit'): perceptron_strategy(data),
	('STDEV', 'OBV'): perceptron_strategy(data),
	('STDEV', 'CMF'): perceptron_strategy(data),
	('STDEV', 'VROC'): perceptron_strategy(data),
	('STDEV', 'MFI'): perceptron_strategy(data),
	('STDEV', 'ADL'): perceptron_strategy(data),
	('STDEV', 'EOM'): perceptron_strategy(data),
	('STDEV', 'Pivot_Points'): perceptron_strategy(data),
	('STDEV', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('STDEV', 'SRL'): perceptron_strategy(data),
	('STDEV', 'Gann_Lines'): perceptron_strategy(data),
	('STDEV', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('STDEV', 'MA_Support_Resistance'): perceptron_strategy(data),
	('STDEV', 'Awesome_Oscillator'): perceptron_strategy(data),
	('KC', 'SMA'): sma_kc_strategy(data),
	('KC', 'EMA'): ema_kc_strategy(data),
	('KC', 'MACD'): macd_kc_strategy(data),
	('KC', 'PSAR'): psar_kc_strategy(data),
	('KC', 'ICHIMOKU'): perceptron_strategy(data),
	('KC', 'SUPERTREND'): perceptron_strategy(data),
	('KC', 'RSI'): perceptron_strategy(data),
	('KC', 'Stochastic'): perceptron_strategy(data),
	('KC', 'CCI'): perceptron_strategy(data),
	('KC', 'ROC'): perceptron_strategy(data),
	('KC', 'WPR'): perceptron_strategy(data),
	('KC', 'MACD_Hist'): perceptron_strategy(data),
	('KC', 'BBANDS'): perceptron_strategy(data),
	('KC', 'ATR'): perceptron_strategy(data),
	('KC', 'STDEV'): perceptron_strategy(data),
	('KC', 'KC'): perceptron_strategy(data),
	('KC', 'Donchian'): perceptron_strategy(data),
	('KC', 'Chandelier_Exit'): perceptron_strategy(data),
	('KC', 'OBV'): perceptron_strategy(data),
	('KC', 'CMF'): perceptron_strategy(data),
	('KC', 'VROC'): perceptron_strategy(data),
	('KC', 'MFI'): perceptron_strategy(data),
	('KC', 'ADL'): perceptron_strategy(data),
	('KC', 'EOM'): perceptron_strategy(data),
	('KC', 'Pivot_Points'): perceptron_strategy(data),
	('KC', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('KC', 'SRL'): perceptron_strategy(data),
	('KC', 'Gann_Lines'): perceptron_strategy(data),
	('KC', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('KC', 'MA_Support_Resistance'): perceptron_strategy(data),
	('KC', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Donchian', 'SMA'): sma_donchian_strategy(data),
	('Donchian', 'EMA'): ema_donchian_strategy(data),
	('Donchian', 'MACD'): macd_donchian_strategy(data),
	('Donchian', 'PSAR'): psar_donchian_strategy(data),
	('Donchian', 'ICHIMOKU'): perceptron_strategy(data),
	('Donchian', 'SUPERTREND'): perceptron_strategy(data),
	('Donchian', 'RSI'): perceptron_strategy(data),
	('Donchian', 'Stochastic'): perceptron_strategy(data),
	('Donchian', 'CCI'): perceptron_strategy(data),
	('Donchian', 'ROC'): perceptron_strategy(data),
	('Donchian', 'WPR'): perceptron_strategy(data),
	('Donchian', 'MACD_Hist'): perceptron_strategy(data),
	('Donchian', 'BBANDS'): perceptron_strategy(data),
	('Donchian', 'ATR'): perceptron_strategy(data),
	('Donchian', 'STDEV'): perceptron_strategy(data),
	('Donchian', 'KC'): perceptron_strategy(data),
	('Donchian', 'Donchian'): perceptron_strategy(data),
	('Donchian', 'Chandelier_Exit'): perceptron_strategy(data),
	('Donchian', 'OBV'): perceptron_strategy(data),
	('Donchian', 'CMF'): perceptron_strategy(data),
	('Donchian', 'VROC'): perceptron_strategy(data),
	('Donchian', 'MFI'): perceptron_strategy(data),
	('Donchian', 'ADL'): perceptron_strategy(data),
	('Donchian', 'EOM'): perceptron_strategy(data),
	('Donchian', 'Pivot_Points'): perceptron_strategy(data),
	('Donchian', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Donchian', 'SRL'): perceptron_strategy(data),
	('Donchian', 'Gann_Lines'): perceptron_strategy(data),
	('Donchian', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Donchian', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Donchian', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Chandelier_Exit', 'SMA'): sma_chandelier_exit_strategy(data),
	('Chandelier_Exit', 'EMA'): ema_chandelier_exit_strategy(data),
	('Chandelier_Exit', 'MACD'): macd_chandelier_exit_strategy(data),
	('Chandelier_Exit', 'PSAR'): psar_chandelier_exit_strategy(data),
	('Chandelier_Exit', 'ICHIMOKU'): perceptron_strategy(data),
	('Chandelier_Exit', 'SUPERTREND'): perceptron_strategy(data),
	('Chandelier_Exit', 'RSI'): perceptron_strategy(data),
	('Chandelier_Exit', 'Stochastic'): perceptron_strategy(data),
	('Chandelier_Exit', 'CCI'): perceptron_strategy(data),
	('Chandelier_Exit', 'ROC'): perceptron_strategy(data),
	('Chandelier_Exit', 'WPR'): perceptron_strategy(data),
	('Chandelier_Exit', 'MACD_Hist'): perceptron_strategy(data),
	('Chandelier_Exit', 'BBANDS'): perceptron_strategy(data),
	('Chandelier_Exit', 'ATR'): perceptron_strategy(data),
	('Chandelier_Exit', 'STDEV'): perceptron_strategy(data),
	('Chandelier_Exit', 'KC'): perceptron_strategy(data),
	('Chandelier_Exit', 'Donchian'): perceptron_strategy(data),
	('Chandelier_Exit', 'Chandelier_Exit'): perceptron_strategy(data),
	('Chandelier_Exit', 'OBV'): perceptron_strategy(data),
	('Chandelier_Exit', 'CMF'): perceptron_strategy(data),
	('Chandelier_Exit', 'VROC'): perceptron_strategy(data),
	('Chandelier_Exit', 'MFI'): perceptron_strategy(data),
	('Chandelier_Exit', 'ADL'): perceptron_strategy(data),
	('Chandelier_Exit', 'EOM'): perceptron_strategy(data),
	('Chandelier_Exit', 'Pivot_Points'): perceptron_strategy(data),
	('Chandelier_Exit', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Chandelier_Exit', 'SRL'): perceptron_strategy(data),
	('Chandelier_Exit', 'Gann_Lines'): perceptron_strategy(data),
	('Chandelier_Exit', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Chandelier_Exit', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Chandelier_Exit', 'Awesome_Oscillator'): perceptron_strategy(data),
	('OBV', 'SMA'): sma_obv_strategy(data),
	('OBV', 'EMA'): ema_obv_strategy(data),
	('OBV', 'MACD'): macd_obv_strategy(data),
	('OBV', 'PSAR'): psar_obv_strategy(data),
	('OBV', 'ICHIMOKU'): perceptron_strategy(data),
	('OBV', 'SUPERTREND'): perceptron_strategy(data),
	('OBV', 'RSI'): perceptron_strategy(data),
	('OBV', 'Stochastic'): perceptron_strategy(data),
	('OBV', 'CCI'): perceptron_strategy(data),
	('OBV', 'ROC'): perceptron_strategy(data),
	('OBV', 'WPR'): perceptron_strategy(data),
	('OBV', 'MACD_Hist'): perceptron_strategy(data),
	('OBV', 'BBANDS'): perceptron_strategy(data),
	('OBV', 'ATR'): perceptron_strategy(data),
	('OBV', 'STDEV'): perceptron_strategy(data),
	('OBV', 'KC'): perceptron_strategy(data),
	('OBV', 'Donchian'): perceptron_strategy(data),
	('OBV', 'Chandelier_Exit'): perceptron_strategy(data),
	('OBV', 'OBV'): perceptron_strategy(data),
	('OBV', 'CMF'): perceptron_strategy(data),
	('OBV', 'VROC'): perceptron_strategy(data),
	('OBV', 'MFI'): perceptron_strategy(data),
	('OBV', 'ADL'): perceptron_strategy(data),
	('OBV', 'EOM'): perceptron_strategy(data),
	('OBV', 'Pivot_Points'): perceptron_strategy(data),
	('OBV', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('OBV', 'SRL'): perceptron_strategy(data),
	('OBV', 'Gann_Lines'): perceptron_strategy(data),
	('OBV', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('OBV', 'MA_Support_Resistance'): perceptron_strategy(data),
	('OBV', 'Awesome_Oscillator'): perceptron_strategy(data),
	('CMF', 'SMA'): sma_cmf_strategy(data),
	('CMF', 'EMA'): ema_cmf_strategy(data),
	('CMF', 'MACD'): macd_cmf_strategy(data),
	('CMF', 'PSAR'): psar_cmf_strategy(data),
	('CMF', 'ICHIMOKU'): perceptron_strategy(data),
	('CMF', 'SUPERTREND'): perceptron_strategy(data),
	('CMF', 'RSI'): perceptron_strategy(data),
	('CMF', 'Stochastic'): perceptron_strategy(data),
	('CMF', 'CCI'): perceptron_strategy(data),
	('CMF', 'ROC'): perceptron_strategy(data),
	('CMF', 'WPR'): perceptron_strategy(data),
	('CMF', 'MACD_Hist'): perceptron_strategy(data),
	('CMF', 'BBANDS'): perceptron_strategy(data),
	('CMF', 'ATR'): perceptron_strategy(data),
	('CMF', 'STDEV'): perceptron_strategy(data),
	('CMF', 'KC'): perceptron_strategy(data),
	('CMF', 'Donchian'): perceptron_strategy(data),
	('CMF', 'Chandelier_Exit'): perceptron_strategy(data),
	('CMF', 'OBV'): perceptron_strategy(data),
	('CMF', 'CMF'): perceptron_strategy(data),
	('CMF', 'VROC'): perceptron_strategy(data),
	('CMF', 'MFI'): perceptron_strategy(data),
	('CMF', 'ADL'): perceptron_strategy(data),
	('CMF', 'EOM'): perceptron_strategy(data),
	('CMF', 'Pivot_Points'): perceptron_strategy(data),
	('CMF', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('CMF', 'SRL'): perceptron_strategy(data),
	('CMF', 'Gann_Lines'): perceptron_strategy(data),
	('CMF', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('CMF', 'MA_Support_Resistance'): perceptron_strategy(data),
	('CMF', 'Awesome_Oscillator'): perceptron_strategy(data),
	('VROC', 'SMA'): sma_vroc_strategy(data),
	('VROC', 'EMA'): ema_vroc_strategy(data),
	('VROC', 'MACD'): macd_vroc_strategy(data),
	('VROC', 'PSAR'): psar_vroc_strategy(data),
	('VROC', 'ICHIMOKU'): perceptron_strategy(data),
	('VROC', 'SUPERTREND'): perceptron_strategy(data),
	('VROC', 'RSI'): perceptron_strategy(data),
	('VROC', 'Stochastic'): perceptron_strategy(data),
	('VROC', 'CCI'): perceptron_strategy(data),
	('VROC', 'ROC'): perceptron_strategy(data),
	('VROC', 'WPR'): perceptron_strategy(data),
	('VROC', 'MACD_Hist'): perceptron_strategy(data),
	('VROC', 'BBANDS'): perceptron_strategy(data),
	('VROC', 'ATR'): perceptron_strategy(data),
	('VROC', 'STDEV'): perceptron_strategy(data),
	('VROC', 'KC'): perceptron_strategy(data),
	('VROC', 'Donchian'): perceptron_strategy(data),
	('VROC', 'Chandelier_Exit'): perceptron_strategy(data),
	('VROC', 'OBV'): perceptron_strategy(data),
	('VROC', 'CMF'): perceptron_strategy(data),
	('VROC', 'VROC'): perceptron_strategy(data),
	('VROC', 'MFI'): perceptron_strategy(data),
	('VROC', 'ADL'): perceptron_strategy(data),
	('VROC', 'EOM'): perceptron_strategy(data),
	('VROC', 'Pivot_Points'): perceptron_strategy(data),
	('VROC', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('VROC', 'SRL'): perceptron_strategy(data),
	('VROC', 'Gann_Lines'): perceptron_strategy(data),
	('VROC', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('VROC', 'MA_Support_Resistance'): perceptron_strategy(data),
	('VROC', 'Awesome_Oscillator'): perceptron_strategy(data),
	('MFI', 'SMA'): sma_mfi_strategy(data),
	('MFI', 'EMA'): ema_mfi_strategy(data),
	('MFI', 'MACD'): macd_mfi_strategy(data),
	('MFI', 'PSAR'): psar_mfi_strategy(data),
	('MFI', 'ICHIMOKU'): perceptron_strategy(data),
	('MFI', 'SUPERTREND'): perceptron_strategy(data),
	('MFI', 'RSI'): perceptron_strategy(data),
	('MFI', 'Stochastic'): perceptron_strategy(data),
	('MFI', 'CCI'): perceptron_strategy(data),
	('MFI', 'ROC'): perceptron_strategy(data),
	('MFI', 'WPR'): perceptron_strategy(data),
	('MFI', 'MACD_Hist'): perceptron_strategy(data),
	('MFI', 'BBANDS'): perceptron_strategy(data),
	('MFI', 'ATR'): perceptron_strategy(data),
	('MFI', 'STDEV'): perceptron_strategy(data),
	('MFI', 'KC'): perceptron_strategy(data),
	('MFI', 'Donchian'): perceptron_strategy(data),
	('MFI', 'Chandelier_Exit'): perceptron_strategy(data),
	('MFI', 'OBV'): perceptron_strategy(data),
	('MFI', 'CMF'): perceptron_strategy(data),
	('MFI', 'VROC'): perceptron_strategy(data),
	('MFI', 'MFI'): perceptron_strategy(data),
	('MFI', 'ADL'): perceptron_strategy(data),
	('MFI', 'EOM'): perceptron_strategy(data),
	('MFI', 'Pivot_Points'): perceptron_strategy(data),
	('MFI', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('MFI', 'SRL'): perceptron_strategy(data),
	('MFI', 'Gann_Lines'): perceptron_strategy(data),
	('MFI', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('MFI', 'MA_Support_Resistance'): perceptron_strategy(data),
	('MFI', 'Awesome_Oscillator'): perceptron_strategy(data),
	('ADL', 'SMA'): sma_adl_strategy(data),
	('ADL', 'EMA'): ema_adl_strategy(data),
	('ADL', 'MACD'): macd_adl_strategy(data),
	('ADL', 'PSAR'): psar_adl_strategy(data),
	('ADL', 'ICHIMOKU'): perceptron_strategy(data),
	('ADL', 'SUPERTREND'): perceptron_strategy(data),
	('ADL', 'RSI'): perceptron_strategy(data),
	('ADL', 'Stochastic'): perceptron_strategy(data),
	('ADL', 'CCI'): perceptron_strategy(data),
	('ADL', 'ROC'): perceptron_strategy(data),
	('ADL', 'WPR'): perceptron_strategy(data),
	('ADL', 'MACD_Hist'): perceptron_strategy(data),
	('ADL', 'BBANDS'): perceptron_strategy(data),
	('ADL', 'ATR'): perceptron_strategy(data),
	('ADL', 'STDEV'): perceptron_strategy(data),
	('ADL', 'KC'): perceptron_strategy(data),
	('ADL', 'Donchian'): perceptron_strategy(data),
	('ADL', 'Chandelier_Exit'): perceptron_strategy(data),
	('ADL', 'OBV'): perceptron_strategy(data),
	('ADL', 'CMF'): perceptron_strategy(data),
	('ADL', 'VROC'): perceptron_strategy(data),
	('ADL', 'MFI'): perceptron_strategy(data),
	('ADL', 'ADL'): perceptron_strategy(data),
	('ADL', 'EOM'): perceptron_strategy(data),
	('ADL', 'Pivot_Points'): perceptron_strategy(data),
	('ADL', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('ADL', 'SRL'): perceptron_strategy(data),
	('ADL', 'Gann_Lines'): perceptron_strategy(data),
	('ADL', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('ADL', 'MA_Support_Resistance'): perceptron_strategy(data),
	('ADL', 'Awesome_Oscillator'): perceptron_strategy(data),
	('EOM', 'SMA'): sma_eom_strategy(data),
	('EOM', 'EMA'): ema_eom_strategy(data),
	('EOM', 'MACD'): macd_eom_strategy(data),
	('EOM', 'PSAR'): psar_eom_strategy(data),
	('EOM', 'ICHIMOKU'): perceptron_strategy(data),
	('EOM', 'SUPERTREND'): perceptron_strategy(data),
	('EOM', 'RSI'): perceptron_strategy(data),
	('EOM', 'Stochastic'): perceptron_strategy(data),
	('EOM', 'CCI'): perceptron_strategy(data),
	('EOM', 'ROC'): perceptron_strategy(data),
	('EOM', 'WPR'): perceptron_strategy(data),
	('EOM', 'MACD_Hist'): perceptron_strategy(data),
	('EOM', 'BBANDS'): perceptron_strategy(data),
	('EOM', 'ATR'): perceptron_strategy(data),
	('EOM', 'STDEV'): perceptron_strategy(data),
	('EOM', 'KC'): perceptron_strategy(data),
	('EOM', 'Donchian'): perceptron_strategy(data),
	('EOM', 'Chandelier_Exit'): perceptron_strategy(data),
	('EOM', 'OBV'): perceptron_strategy(data),
	('EOM', 'CMF'): perceptron_strategy(data),
	('EOM', 'VROC'): perceptron_strategy(data),
	('EOM', 'MFI'): perceptron_strategy(data),
	('EOM', 'ADL'): perceptron_strategy(data),
	('EOM', 'EOM'): perceptron_strategy(data),
	('EOM', 'Pivot_Points'): perceptron_strategy(data),
	('EOM', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('EOM', 'SRL'): perceptron_strategy(data),
	('EOM', 'Gann_Lines'): perceptron_strategy(data),
	('EOM', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('EOM', 'MA_Support_Resistance'): perceptron_strategy(data),
	('EOM', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Pivot_Points', 'SMA'): sma_pivot_points_strategy(data),
	('Pivot_Points', 'EMA'): ema_pivot_points_strategy(data),
	('Pivot_Points', 'MACD'): macd_pivot_points_strategy(data),
	('Pivot_Points', 'PSAR'): psar_pivot_strategy(data),
	('Pivot_Points', 'ICHIMOKU'): perceptron_strategy(data),
	('Pivot_Points', 'SUPERTREND'): perceptron_strategy(data),
	('Pivot_Points', 'RSI'): perceptron_strategy(data),
	('Pivot_Points', 'Stochastic'): perceptron_strategy(data),
	('Pivot_Points', 'CCI'): perceptron_strategy(data),
	('Pivot_Points', 'ROC'): perceptron_strategy(data),
	('Pivot_Points', 'WPR'): perceptron_strategy(data),
	('Pivot_Points', 'MACD_Hist'): perceptron_strategy(data),
	('Pivot_Points', 'BBANDS'): perceptron_strategy(data),
	('Pivot_Points', 'ATR'): perceptron_strategy(data),
	('Pivot_Points', 'STDEV'): perceptron_strategy(data),
	('Pivot_Points', 'KC'): perceptron_strategy(data),
	('Pivot_Points', 'Donchian'): perceptron_strategy(data),
	('Pivot_Points', 'Chandelier_Exit'): perceptron_strategy(data),
	('Pivot_Points', 'OBV'): perceptron_strategy(data),
	('Pivot_Points', 'CMF'): perceptron_strategy(data),
	('Pivot_Points', 'VROC'): perceptron_strategy(data),
	('Pivot_Points', 'MFI'): perceptron_strategy(data),
	('Pivot_Points', 'ADL'): perceptron_strategy(data),
	('Pivot_Points', 'EOM'): perceptron_strategy(data),
	('Pivot_Points', 'Pivot_Points'): perceptron_strategy(data),
	('Pivot_Points', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Pivot_Points', 'SRL'): perceptron_strategy(data),
	('Pivot_Points', 'Gann_Lines'): perceptron_strategy(data),
	('Pivot_Points', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Pivot_Points', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Pivot_Points', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'SMA'): sma_fibonacci_retracement_strategy(data),
	('Fibonacci_Retracement', 'EMA'): ema_fibonacci_strategy(data),
	('Fibonacci_Retracement', 'MACD'): macd_fibonacci_strategy(data),
	('Fibonacci_Retracement', 'PSAR'): psar_fibonacci_strategy(data),
	('Fibonacci_Retracement', 'ICHIMOKU'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'SUPERTREND'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'RSI'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Stochastic'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'CCI'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'ROC'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'WPR'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'MACD_Hist'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'BBANDS'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'ATR'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'STDEV'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'KC'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Donchian'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Chandelier_Exit'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'OBV'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'CMF'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'VROC'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'MFI'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'ADL'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'EOM'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Pivot_Points'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'SRL'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Gann_Lines'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'Awesome_Oscillator'): perceptron_strategy(data),
	('SRL', 'SMA'): sma_srl_strategy(data),
	('SRL', 'EMA'): ema_srl_strategy(data),
	('SRL', 'MACD'): macd_srl_strategy(data),
	('SRL', 'PSAR'): psar_srl_strategy(data),
	('SRL', 'ICHIMOKU'): perceptron_strategy(data),
	('SRL', 'SUPERTREND'): perceptron_strategy(data),
	('SRL', 'RSI'): perceptron_strategy(data),
	('SRL', 'Stochastic'): perceptron_strategy(data),
	('SRL', 'CCI'): perceptron_strategy(data),
	('SRL', 'ROC'): perceptron_strategy(data),
	('SRL', 'WPR'): perceptron_strategy(data),
	('SRL', 'MACD_Hist'): perceptron_strategy(data),
	('SRL', 'BBANDS'): perceptron_strategy(data),
	('SRL', 'ATR'): perceptron_strategy(data),
	('SRL', 'STDEV'): perceptron_strategy(data),
	('SRL', 'KC'): perceptron_strategy(data),
	('SRL', 'Donchian'): perceptron_strategy(data),
	('SRL', 'Chandelier_Exit'): perceptron_strategy(data),
	('SRL', 'OBV'): perceptron_strategy(data),
	('SRL', 'CMF'): perceptron_strategy(data),
	('SRL', 'VROC'): perceptron_strategy(data),
	('SRL', 'MFI'): perceptron_strategy(data),
	('SRL', 'ADL'): perceptron_strategy(data),
	('SRL', 'EOM'): perceptron_strategy(data),
	('SRL', 'Pivot_Points'): perceptron_strategy(data),
	('SRL', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('SRL', 'SRL'): perceptron_strategy(data),
	('SRL', 'Gann_Lines'): perceptron_strategy(data),
	('SRL', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('SRL', 'MA_Support_Resistance'): perceptron_strategy(data),
	('SRL', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Gann_Lines', 'SMA'): sma_gann_lines_strategy(data),
	('Gann_Lines', 'EMA'): ema_gann_lines_strategy(data),
	('Gann_Lines', 'MACD'): macd_gann_lines_strategy(data),
	('Gann_Lines', 'PSAR'): psar_gann_strategy(data),
	('Gann_Lines', 'ICHIMOKU'): perceptron_strategy(data),
	('Gann_Lines', 'SUPERTREND'): perceptron_strategy(data),
	('Gann_Lines', 'RSI'): perceptron_strategy(data),
	('Gann_Lines', 'Stochastic'): perceptron_strategy(data),
	('Gann_Lines', 'CCI'): perceptron_strategy(data),
	('Gann_Lines', 'ROC'): perceptron_strategy(data),
	('Gann_Lines', 'WPR'): perceptron_strategy(data),
	('Gann_Lines', 'MACD_Hist'): perceptron_strategy(data),
	('Gann_Lines', 'BBANDS'): perceptron_strategy(data),
	('Gann_Lines', 'ATR'): perceptron_strategy(data),
	('Gann_Lines', 'STDEV'): perceptron_strategy(data),
	('Gann_Lines', 'KC'): perceptron_strategy(data),
	('Gann_Lines', 'Donchian'): perceptron_strategy(data),
	('Gann_Lines', 'Chandelier_Exit'): perceptron_strategy(data),
	('Gann_Lines', 'OBV'): perceptron_strategy(data),
	('Gann_Lines', 'CMF'): perceptron_strategy(data),
	('Gann_Lines', 'VROC'): perceptron_strategy(data),
	('Gann_Lines', 'MFI'): perceptron_strategy(data),
	('Gann_Lines', 'ADL'): perceptron_strategy(data),
	('Gann_Lines', 'EOM'): perceptron_strategy(data),
	('Gann_Lines', 'Pivot_Points'): perceptron_strategy(data),
	('Gann_Lines', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Gann_Lines', 'SRL'): perceptron_strategy(data),
	('Gann_Lines', 'Gann_Lines'): perceptron_strategy(data),
	('Gann_Lines', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Gann_Lines', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Gann_Lines', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'SMA'): sma_andrews_pitchfork_strategy(data),
	('Andrews_Pitchfork', 'EMA'): ema_andrews_pitchfork_strategy(data),
	('Andrews_Pitchfork', 'MACD'): macd_andrews_pitchfork_strategy(data),
	('Andrews_Pitchfork', 'PSAR'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'ICHIMOKU'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'SUPERTREND'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'RSI'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Stochastic'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'CCI'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'ROC'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'WPR'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'MACD_Hist'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'BBANDS'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'ATR'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'STDEV'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'KC'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Donchian'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Chandelier_Exit'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'OBV'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'CMF'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'VROC'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'MFI'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'ADL'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'EOM'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Pivot_Points'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'SRL'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Gann_Lines'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'Awesome_Oscillator'): perceptron_strategy(data),
	('MA_Support_Resistance', 'SMA'): sma_ma_support_resistance_strategy(data),
	('MA_Support_Resistance', 'EMA'): ema_sr_strategy(data),
	('MA_Support_Resistance', 'MACD'): macd_ma_sr_strategy(data),
	('MA_Support_Resistance', 'PSAR'): perceptron_strategy(data),
	('MA_Support_Resistance', 'ICHIMOKU'): perceptron_strategy(data),
	('MA_Support_Resistance', 'SUPERTREND'): perceptron_strategy(data),
	('MA_Support_Resistance', 'RSI'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Stochastic'): perceptron_strategy(data),
	('MA_Support_Resistance', 'CCI'): perceptron_strategy(data),
	('MA_Support_Resistance', 'ROC'): perceptron_strategy(data),
	('MA_Support_Resistance', 'WPR'): perceptron_strategy(data),
	('MA_Support_Resistance', 'MACD_Hist'): perceptron_strategy(data),
	('MA_Support_Resistance', 'BBANDS'): perceptron_strategy(data),
	('MA_Support_Resistance', 'ATR'): perceptron_strategy(data),
	('MA_Support_Resistance', 'STDEV'): perceptron_strategy(data),
	('MA_Support_Resistance', 'KC'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Donchian'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Chandelier_Exit'): perceptron_strategy(data),
	('MA_Support_Resistance', 'OBV'): perceptron_strategy(data),
	('MA_Support_Resistance', 'CMF'): perceptron_strategy(data),
	('MA_Support_Resistance', 'VROC'): perceptron_strategy(data),
	('MA_Support_Resistance', 'MFI'): perceptron_strategy(data),
	('MA_Support_Resistance', 'ADL'): perceptron_strategy(data),
	('MA_Support_Resistance', 'EOM'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Pivot_Points'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('MA_Support_Resistance', 'SRL'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Gann_Lines'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('MA_Support_Resistance', 'MA_Support_Resistance'): perceptron_strategy(data),
	('MA_Support_Resistance', 'Awesome_Oscillator'): perceptron_strategy(data),
	('Awesome_Oscillator', 'SMA'): sma_awesome_oscillator_strategy(data),
	('Awesome_Oscillator', 'EMA'): ema_awesome_oscillator_strategy(data),
	('Awesome_Oscillator', 'MACD'): macd_awesome_oscillator_strategy(data),
	('Awesome_Oscillator', 'PSAR'): perceptron_strategy(data),
	('Awesome_Oscillator', 'ICHIMOKU'): perceptron_strategy(data),
	('Awesome_Oscillator', 'SUPERTREND'): perceptron_strategy(data),
	('Awesome_Oscillator', 'RSI'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Stochastic'): perceptron_strategy(data),
	('Awesome_Oscillator', 'CCI'): perceptron_strategy(data),
	('Awesome_Oscillator', 'ROC'): perceptron_strategy(data),
	('Awesome_Oscillator', 'WPR'): perceptron_strategy(data),
	('Awesome_Oscillator', 'MACD_Hist'): perceptron_strategy(data),
	('Awesome_Oscillator', 'BBANDS'): perceptron_strategy(data),
	('Awesome_Oscillator', 'ATR'): perceptron_strategy(data),
	('Awesome_Oscillator', 'STDEV'): perceptron_strategy(data),
	('Awesome_Oscillator', 'KC'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Donchian'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Chandelier_Exit'): perceptron_strategy(data),
	('Awesome_Oscillator', 'OBV'): perceptron_strategy(data),
	('Awesome_Oscillator', 'CMF'): perceptron_strategy(data),
	('Awesome_Oscillator', 'VROC'): perceptron_strategy(data),
	('Awesome_Oscillator', 'MFI'): perceptron_strategy(data),
	('Awesome_Oscillator', 'ADL'): perceptron_strategy(data),
	('Awesome_Oscillator', 'EOM'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Pivot_Points'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('Awesome_Oscillator', 'SRL'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Gann_Lines'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('Awesome_Oscillator', 'MA_Support_Resistance'): perceptron_strategy(data),
	('Awesome_Oscillator', 'Awesome_Oscillator'): perceptron_strategy(data)
}

# Define a function to calculate the strategy for a given indicator combination
def calculate_strategy(indicator1, indicator2):
    try:
        return strategies[(indicator1, indicator2)]
    except KeyError:
        return strategies[('ANY', 'ANY')]

# Generate a 36x36 matrix of strategies for all indicator combinations
matrix = []
for i1 in indicators:
    row = []
    for i2 in indicators:
        row.append(calculate_strategy(i1, i2))
    matrix.append(row)

# data.fillna(0,inplace=True)
# data.to_csv('matrix.csv')
# print(data)
data = np.array(matrix)


# Define a color map for the heatmap
cmap = plt.cm.get_cmap('RdYlGn', 3)

# Create the heatmap plot
fig, ax = plt.subplots()
im = ax.imshow(data, cmap=cmap)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, ticks=[-1, 0, 1], orientation='vertical')
cbar.ax.set_yticklabels(['Sell', 'Hold', 'Buy'])

# Set the tick labels and rotation for the x and y axes
ax.set_xticks(np.arange(len(indicators)))
ax.set_yticks(np.arange(len(indicators)))
ax.set_xticklabels(indicators, rotation=90, ha='right')
ax.set_yticklabels(indicators)

# Loop over data dimensions and create text annotations.
for i in range(len(indicators)):
    for j in range(len(indicators)):
        text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="black")

# Add lines to separate the indicator groups
for i, group in enumerate(indicators_groups.keys()):
    if i >= 0:
        start = sum([len(x) for x in list(indicators_groups.values())[:i]])
        plt.axhline(start-0.5, color='black', lw=2)
        plt.axvline(start-0.5, color='black', lw=2)
        ax.text(len(indicators), -1+start+len(indicators_groups[group])/2, group, ha='center', va='center', rotation=270, fontsize=10)

        

# Set the title and show the plot
plt.title("Indicator Heatmap")
plt.show()
data = pd.read_csv("EURUSD=X.csv", index_col=0)
signal = psar_gann_strategy(data)

# Print signal value
print(signal)
# print(strategies)
