import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# Load data from CSV file
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Create a figure and axes for subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot Candlestick Chart
mpf.plot(data, ax=axes[0], type='candle', volume=False,style='yahoo')
axes[0].set_title('Candlestick Chart')

# Plot Point and Figure Chart
mpf.plot(data, ax=axes[1], type='pnf', volume=False,style='yahoo')
axes[1].set_title('Point and Figure Chart')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
