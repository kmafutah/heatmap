import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import ta.trend as tr
import ta.volatility as vo
import ta.momentum as mo

from ta.volume import ChaikinMoneyFlowIndicator
from ta.momentum import ROCIndicator
from ta.volume import money_flow_index
from ta.volume import AccDistIndexIndicator
from ta.volume import EaseOfMovementIndicator
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import SMAIndicator


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
    # print(sdf)
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
    # df.columns = map(str.upper, data.columns)

    sma = df['adj close'].rolling(window=20).mean()
    obv = [0]
    for i in range(1, len(df)):
        if df['adj close'][i] > df['adj close'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['adj close'][i] < df['adj close'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Apply SMA-OBV strategy
    buy_signal = df['adj close'].iloc[-1] > sma.iloc[-1] and df['OBV'].iloc[-1] > df['OBV'].iloc[-2]
    sell_signal = df['adj close'].iloc[-1] < sma.iloc[-1] and df['OBV'].iloc[-1] < df['OBV'].iloc[-2]

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
    print(df)    
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


# Define the labels for the heatmap
trend_indicators = ['SMA', 'EMA', 'MACD', 'PSAR', 'ICHIMOKU', 'SUPERTREND']
momentum_indicators = ['RSI', 'Stochastic', 'CCI', 'ROC', 'WPR', 'MACD_Hist']
volatility_indicators = ['BBANDS', 'ATR', 'STDEV', 'KC', 'Donchian', 'Chandelier_Exit']
volume_indicators = ['OBV', 'CMF', 'VROC', 'MFI', 'ADL', 'EOM']
support_resistance_indicators = ['Pivot_Points', 'Fibonacci_Retracement', 'SRL', 'Gann_Lines', 'Andrews_Pitchfork', 'MA_Support_Resistance']
oscillator_indicators = ['RSI', 'MACD', 'Stochastic', 'Awesome_Oscillator', 'WPR', 'CCI']

# Define the labels for the indicators
indicators = trend_indicators + momentum_indicators + volatility_indicators + volume_indicators + support_resistance_indicators + oscillator_indicators
# data = pd.read_csv("EURUSD=X.csv", index_col=0)

# Define the stock symbol and time period to download
# symbol = "EURGBP=X"
symbol = "EURUSD=X"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=1440)).strftime('%Y-%m-%d')

# Download the historical data from Yahoo Finance
data = yf.download(symbol, start=start_date, end=end_date)


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
	('EMA', 'ICHIMOKU'): perceptron_strategy(data),
	('EMA', 'SUPERTREND'): perceptron_strategy(data),
	('EMA', 'RSI'): perceptron_strategy(data),
	('EMA', 'Stochastic'): perceptron_strategy(data),
	('EMA', 'CCI'): perceptron_strategy(data),
	('EMA', 'ROC'): perceptron_strategy(data),
	('EMA', 'WPR'): perceptron_strategy(data),
	('EMA', 'MACD_Hist'): perceptron_strategy(data),
	('EMA', 'BBANDS'): perceptron_strategy(data),
	('EMA', 'ATR'): perceptron_strategy(data),
	('EMA', 'STDEV'): perceptron_strategy(data),
	('EMA', 'KC'): perceptron_strategy(data),
	('EMA', 'Donchian'): perceptron_strategy(data),
	('EMA', 'Chandelier_Exit'): perceptron_strategy(data),
	('EMA', 'OBV'): perceptron_strategy(data),
	('EMA', 'CMF'): perceptron_strategy(data),
	('EMA', 'VROC'): perceptron_strategy(data),
	('EMA', 'MFI'): perceptron_strategy(data),
	('EMA', 'ADL'): perceptron_strategy(data),
	('EMA', 'EOM'): perceptron_strategy(data),
	('EMA', 'Pivot_Points'): perceptron_strategy(data),
	('EMA', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('EMA', 'SRL'): perceptron_strategy(data),
	('EMA', 'Gann_Lines'): perceptron_strategy(data),
	('EMA', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('EMA', 'MA_Support_Resistance'): perceptron_strategy(data),
	('EMA', 'Awesome_Oscillator'): perceptron_strategy(data),
	('MACD', 'SMA'): sma_macd_strategy(data),
	('MACD', 'EMA'): ema_macd_strategy(data),
	('MACD', 'MACD'): perceptron_strategy(data),
	('MACD', 'PSAR'): perceptron_strategy(data),
	('MACD', 'ICHIMOKU'): perceptron_strategy(data),
	('MACD', 'SUPERTREND'): perceptron_strategy(data),
	('MACD', 'RSI'): perceptron_strategy(data),
	('MACD', 'Stochastic'): perceptron_strategy(data),
	('MACD', 'CCI'): perceptron_strategy(data),
	('MACD', 'ROC'): perceptron_strategy(data),
	('MACD', 'WPR'): perceptron_strategy(data),
	('MACD', 'MACD_Hist'): perceptron_strategy(data),
	('MACD', 'BBANDS'): perceptron_strategy(data),
	('MACD', 'ATR'): perceptron_strategy(data),
	('MACD', 'STDEV'): perceptron_strategy(data),
	('MACD', 'KC'): perceptron_strategy(data),
	('MACD', 'Donchian'): perceptron_strategy(data),
	('MACD', 'Chandelier_Exit'): perceptron_strategy(data),
	('MACD', 'OBV'): perceptron_strategy(data),
	('MACD', 'CMF'): perceptron_strategy(data),
	('MACD', 'VROC'): perceptron_strategy(data),
	('MACD', 'MFI'): perceptron_strategy(data),
	('MACD', 'ADL'): perceptron_strategy(data),
	('MACD', 'EOM'): perceptron_strategy(data),
	('MACD', 'Pivot_Points'): perceptron_strategy(data),
	('MACD', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('MACD', 'SRL'): perceptron_strategy(data),
	('MACD', 'Gann_Lines'): perceptron_strategy(data),
	('MACD', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('MACD', 'MA_Support_Resistance'): perceptron_strategy(data),
	('MACD', 'Awesome_Oscillator'): perceptron_strategy(data),
	('PSAR', 'SMA'): sma_psar_strategy(data),
	('PSAR', 'EMA'): ema_psar_strategy(data),
	('PSAR', 'MACD'): perceptron_strategy(data),
	('PSAR', 'PSAR'): perceptron_strategy(data),
	('PSAR', 'ICHIMOKU'): perceptron_strategy(data),
	('PSAR', 'SUPERTREND'): perceptron_strategy(data),
	('PSAR', 'RSI'): perceptron_strategy(data),
	('PSAR', 'Stochastic'): perceptron_strategy(data),
	('PSAR', 'CCI'): perceptron_strategy(data),
	('PSAR', 'ROC'): perceptron_strategy(data),
	('PSAR', 'WPR'): perceptron_strategy(data),
	('PSAR', 'MACD_Hist'): perceptron_strategy(data),
	('PSAR', 'BBANDS'): perceptron_strategy(data),
	('PSAR', 'ATR'): perceptron_strategy(data),
	('PSAR', 'STDEV'): perceptron_strategy(data),
	('PSAR', 'KC'): perceptron_strategy(data),
	('PSAR', 'Donchian'): perceptron_strategy(data),
	('PSAR', 'Chandelier_Exit'): perceptron_strategy(data),
	('PSAR', 'OBV'): perceptron_strategy(data),
	('PSAR', 'CMF'): perceptron_strategy(data),
	('PSAR', 'VROC'): perceptron_strategy(data),
	('PSAR', 'MFI'): perceptron_strategy(data),
	('PSAR', 'ADL'): perceptron_strategy(data),
	('PSAR', 'EOM'): perceptron_strategy(data),
	('PSAR', 'Pivot_Points'): perceptron_strategy(data),
	('PSAR', 'Fibonacci_Retracement'): perceptron_strategy(data),
	('PSAR', 'SRL'): perceptron_strategy(data),
	('PSAR', 'Gann_Lines'): perceptron_strategy(data),
	('PSAR', 'Andrews_Pitchfork'): perceptron_strategy(data),
	('PSAR', 'MA_Support_Resistance'): perceptron_strategy(data),
	('PSAR', 'Awesome_Oscillator'): perceptron_strategy(data),
	('ICHIMOKU', 'SMA'): sma_ichimoku_strategy(data),
	('ICHIMOKU', 'EMA'): perceptron_strategy(data),
	('ICHIMOKU', 'MACD'): perceptron_strategy(data),
	('ICHIMOKU', 'PSAR'): perceptron_strategy(data),
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
	('SUPERTREND', 'EMA'): perceptron_strategy(data),
	('SUPERTREND', 'MACD'): perceptron_strategy(data),
	('SUPERTREND', 'PSAR'): perceptron_strategy(data),
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
	('RSI', 'EMA'): perceptron_strategy(data),
	('RSI', 'MACD'): perceptron_strategy(data),
	('RSI', 'PSAR'): perceptron_strategy(data),
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
	('Stochastic', 'EMA'): perceptron_strategy(data),
	('Stochastic', 'MACD'): perceptron_strategy(data),
	('Stochastic', 'PSAR'): perceptron_strategy(data),
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
	('CCI', 'EMA'): perceptron_strategy(data),
	('CCI', 'MACD'): perceptron_strategy(data),
	('CCI', 'PSAR'): perceptron_strategy(data),
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
	('ROC', 'EMA'): perceptron_strategy(data),
	('ROC', 'MACD'): perceptron_strategy(data),
	('ROC', 'PSAR'): perceptron_strategy(data),
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
	('WPR', 'EMA'): perceptron_strategy(data),
	('WPR', 'MACD'): perceptron_strategy(data),
	('WPR', 'PSAR'): perceptron_strategy(data),
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
	('MACD_Hist', 'EMA'): perceptron_strategy(data),
	('MACD_Hist', 'MACD'): perceptron_strategy(data),
	('MACD_Hist', 'PSAR'): perceptron_strategy(data),
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
	('BBANDS', 'EMA'): perceptron_strategy(data),
	('BBANDS', 'MACD'): perceptron_strategy(data),
	('BBANDS', 'PSAR'): perceptron_strategy(data),
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
	('ATR', 'EMA'): perceptron_strategy(data),
	('ATR', 'MACD'): perceptron_strategy(data),
	('ATR', 'PSAR'): perceptron_strategy(data),
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
	('STDEV', 'EMA'): perceptron_strategy(data),
	('STDEV', 'MACD'): perceptron_strategy(data),
	('STDEV', 'PSAR'): perceptron_strategy(data),
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
	('KC', 'EMA'): perceptron_strategy(data),
	('KC', 'MACD'): perceptron_strategy(data),
	('KC', 'PSAR'): perceptron_strategy(data),
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
	('Donchian', 'EMA'): perceptron_strategy(data),
	('Donchian', 'MACD'): perceptron_strategy(data),
	('Donchian', 'PSAR'): perceptron_strategy(data),
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
	('Chandelier_Exit', 'EMA'): perceptron_strategy(data),
	('Chandelier_Exit', 'MACD'): perceptron_strategy(data),
	('Chandelier_Exit', 'PSAR'): perceptron_strategy(data),
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
	('OBV', 'EMA'): perceptron_strategy(data),
	('OBV', 'MACD'): perceptron_strategy(data),
	('OBV', 'PSAR'): perceptron_strategy(data),
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
	('CMF', 'EMA'): perceptron_strategy(data),
	('CMF', 'MACD'): perceptron_strategy(data),
	('CMF', 'PSAR'): perceptron_strategy(data),
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
	('VROC', 'EMA'): perceptron_strategy(data),
	('VROC', 'MACD'): perceptron_strategy(data),
	('VROC', 'PSAR'): perceptron_strategy(data),
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
	('MFI', 'EMA'): perceptron_strategy(data),
	('MFI', 'MACD'): perceptron_strategy(data),
	('MFI', 'PSAR'): perceptron_strategy(data),
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
	('ADL', 'EMA'): perceptron_strategy(data),
	('ADL', 'MACD'): perceptron_strategy(data),
	('ADL', 'PSAR'): perceptron_strategy(data),
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
	('EOM', 'EMA'): perceptron_strategy(data),
	('EOM', 'MACD'): perceptron_strategy(data),
	('EOM', 'PSAR'): perceptron_strategy(data),
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
	('Pivot_Points', 'EMA'): perceptron_strategy(data),
	('Pivot_Points', 'MACD'): perceptron_strategy(data),
	('Pivot_Points', 'PSAR'): perceptron_strategy(data),
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
	('Fibonacci_Retracement', 'EMA'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'MACD'): perceptron_strategy(data),
	('Fibonacci_Retracement', 'PSAR'): perceptron_strategy(data),
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
	('SRL', 'EMA'): perceptron_strategy(data),
	('SRL', 'MACD'): perceptron_strategy(data),
	('SRL', 'PSAR'): perceptron_strategy(data),
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
	('Gann_Lines', 'EMA'): perceptron_strategy(data),
	('Gann_Lines', 'MACD'): perceptron_strategy(data),
	('Gann_Lines', 'PSAR'): perceptron_strategy(data),
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
	('Andrews_Pitchfork', 'EMA'): perceptron_strategy(data),
	('Andrews_Pitchfork', 'MACD'): perceptron_strategy(data),
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
	('MA_Support_Resistance', 'EMA'): perceptron_strategy(data),
	('MA_Support_Resistance', 'MACD'): perceptron_strategy(data),
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
	('Awesome_Oscillator', 'EMA'): perceptron_strategy(data),
	('Awesome_Oscillator', 'MACD'): perceptron_strategy(data),
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
ax.set_xticklabels(indicators, rotation=45, ha='right')
ax.set_yticklabels(indicators)

# Loop over data dimensions and create text annotations.
for i in range(len(indicators)):
    for j in range(len(indicators)):
        text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="black")

# Set the title and show the plot
plt.title("Indicator Heatmap")
plt.show()
data = pd.read_csv("EURUSD=X.csv", index_col=0)
signal = sma_obv_strategy(data)

# Print signal value
print(signal)
# print(strategies)
