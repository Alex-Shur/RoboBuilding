"""
MIT License

Copyright (c) 2026 Alex Shurenberg
"""

import datetime

import pandas as pd
import backtrader_next as bt

from backtrader_next.feeds.pandafeed import PandasData
from R2_Soldiers import R2_Soldiers
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
    cerebro.addstrategy(
        R2_Soldiers,
        sma_filter=True,
        sma_period=150,
        volume_pct=10,
        max_positions=10,
        volatility_cluster=3,
        cluster_lookback=80,
        days_volatility_adaptive=7,
        height_soldiers_vola_percent=80,
        proc_height_take=185,
        proc_height_stop=106,
        printlog=False
    )


    # Run backtest
    cerebro.run()
    print(cerebro.statistics)

    # Plot results
    # cerebro.old_plot(style='candle')
    # cerebro.plot(filename="output_charts.html")
    cerebro.show_report(filename="output_stats.html")
