"""
MIT License

Copyright (c) 2026 Alex Shurenberg
"""

import logging
import datetime

import pandas as pd
import backtrader_next as bt

from backtrader_next.feeds.pandafeed import PandasData
from R1_LinearRegression import R1_LinearRegression
from R_common import stock_names
from bn_quik import QuikStore


# Настройка логирования
logging.basicConfig(
    #level=logging.DEBUG,  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level=logging.INFO,  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Формат сообщений
    datefmt='%Y-%m-%d %H:%M:%S'  # Формат времени
)


if __name__ == '__main__':
    """
    Example usage of the strategy
    """
    ## Для Quik Demo
    store = QuikStore(trade_account_id='NL00111000XX', limit_kind=-1) # for Demo=> limit_kind=-1 
    cerebro = bt.Cerebro(quicknotify=True)
    # Set commission
    cerebro.broker.setcommission(commission=0.0004, leverage=5) # 0.04% per trade

    for name in stock_names:
        symbol = 'QJSIM.'+name  # Для Quik Demo
        data = store.getdata(dataname=symbol, timeframe=bt.TimeFrame.Minutes, compression=30, live_bars=True)
        cerebro.adddata(data, name=name)

    broker = store.getbroker()
    cerebro.setbroker(broker)

    # Add strategy
    cerebro.addstrategy(
        R1_LinearRegression,
        lr_period=180,
        lr_deviation=2.5,
        sma_filter=True,
        sma_period=286,
        cluster_lookback=30,
        volume_pct=10,
        max_positions=10,
        printlog=True,
        live_mode=True)


    # Run live trading
    cerebro.run()

