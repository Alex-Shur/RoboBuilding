"""
MIT License

Copyright (c) 2026 Alex Shurenberg
"""

import datetime

import pandas as pd
import numpy as np
import backtrader_next as bt

from backtrader_next.feeds.pandafeed import PandasData
from R1_LinearRegression import R1_LinearRegression
from R_common import stock_names


if __name__ == '__main__':
    """
    Example usage of the strategy
    """

    # Create a cerebro instance
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000.0)

    # Set commission
    cerebro.broker.setcommission(commission=0.0004, leverage=5) # 0.04% per trade


    for name in stock_names:
        df = pd.read_csv(f"./DATA/{name}_M30.csv.zip", sep=";",  parse_dates=["Datetime"], index_col=0)
        df.index.name = 'Datetime'
        data = PandasData(dataframe=df, timeframe=bt.TimeFrame.Minutes, compression=30, fromdate=datetime.date(2015,1,1))
        cerebro.adddata(data, name=name)


    # Add strategy
    cerebro.optstrategy(
        R1_LinearRegression,
        lr_period=range(20, 301, 10),
        lr_deviation=[round(x,1) for x in np.arange(1.0, 4.1, 0.1)],
        sma_period=range(100, 301, 10),
        cluster_lookback=range(100, 301, 1),
	)

    # Run backtest
    #results = cerebro.run(maxcpus=10)
    results = cerebro.run(maxcpus=1)  # Use single CPU for debugging

    list = []
    for stratrun in results:
        for strat in stratrun:
            v = strat.statistics
            list.append(v)

    df = pd.DataFrame(list)
    print(df.head(5))  # Display first 5 rows of the DataFrame
    print(df.columns)  # Display all columns
    print("\n")

    # Save the Optimization results DataFrame to a CSV file
    df.to_csv('opt_results.csv', index=True, lineterminator='\r\n', sep=';')

    # Sort the DataFrame by 'Sharpe Ratio'
    df_sorted = df.sort_values(by='Sharpe Ratio', ascending=False)

    # Display top 5 strategies by Sharpe Ratio
    s = df_sorted.head(5)[['lr_period', 'lr_deviation', 'sma_period', 'cluster_lookback', 'Sharpe Ratio', 'Cum Return [%]', 'Return (Ann.) [%]', 'Max. Drawdown [%]']].to_string(index=False)
    print(s)


