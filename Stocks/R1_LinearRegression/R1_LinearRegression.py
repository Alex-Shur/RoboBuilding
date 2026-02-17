"""
MIT License

Copyright (c) 2026 Alex Shurenberg

== Linear Regression Channel with Volatility Filter ==

Based on OSEngine C# Robot: AlgoStart1LinearRegression

Strategy Description:
- Trend following strategy based on Linear Regression Channel
- Uses volatility clustering for instrument selection
- Entry: Price closes above upper Linear Regression Channel
- Exit: Price closes below lower Linear Regression Channel
- Optional SMA filter to confirm trend direction

Original C# version  Author: AlexWan (OsEngine)
"""

from collections import defaultdict
import datetime

import pandas as pd
import backtrader_next as bt
import numpy as np
from datetime import time

from indicators import VolatilityStageClusters, LinearRegressionChannel_MAD
from R_common import stock_names


class R1_LinearRegression(bt.Strategy):
    """
    Linear Regression Channel Strategy with Volatility Filtering
    
    Parameters:
    - lr_period: Linear Regression period (default: 180)
    - lr_deviation: Standard deviation multiplier (default: 2.4)
    - sma_filter: Enable SMA filter (default: True)
    - sma_period: SMA filter period (default: 170)
    - volume_pct: Position size as % of portfolio (default: 10)
    - max_positions: Maximum concurrent positions (default: 10)
    - volatility_cluster: Trade only this volatility cluster 1-3 (default: 1)
    - cluster_lookback: Period for volatility clustering (default: 30)
    """
    
    params = dict(
        # Indicator parameters
        lr_period=180,    # 20 - 300 #10
        lr_deviation=2.4, # 1  - 4 #0.1
        sma_filter=True,
        sma_period=170,   # 100 - 300 #10
        
        # Position sizing
        volume_pct=10,    # 1 - 50  #4
        max_positions=10, # 1 - 50  #4
        
        # Volatility clustering
        volatility_cluster=1,  # 1 - 3
        cluster_lookback=30,   # 10 - 300  #1
        
        # Trading hours (Moscow Exchange example)
        trade_start_tm = time(10, 5),
        trade_end_tm = time(18, 0),
        
        # Days to trade
        trade_weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        
        # Order execution
        iceberg_count=1,
        
        # Debug
        printlog=False,
        
        # Set to True for live trading (disables some backtesting features)
        live_mode=False
    )

    def __init__(self):
        # Keep track of pending orders and positions
        self.order = None
        self.position_opened = False

        # Volatility clusters manager
        self.volatility_clusters = VolatilityStageClusters(
                            lookback=self.p.cluster_lookback,
                            one_percent=33.0,    # Lowest volatility
                            two_percent=33.0,     # Medium volatility
                            three_percent=34.0    # Highest volatility
                            )
        self.last_time_set_clusters = None
        
        # Dictionary to store indicators for each data feed
        self.inds = {}
        self.stocks = defaultdict(lambda: bt.DataBase.UNKNOWN)
        for name in stock_names:
            d = self.getdatabyname(name)
            if d is None:
                continue
            self.stocks[name] = bt.DataBase.UNKNOWN

            self.inds[name] = {}
            
            # Linear Regression Channel
            self.inds[name]['lr_channel'] = LinearRegressionChannel_MAD(
                d.close,
                period=self.p.lr_period,
                up_deviation=self.p.lr_deviation,
                down_deviation=self.p.lr_deviation
            )
            
            # SMA Filter (optional)
            if self.p.sma_filter:
                self.inds[name]['sma'] = bt.indicators.SMA(
                    d.close,
                    period=self.p.sma_period
                )

    def log(self, txt, dt=None, doprint=False):
        """Logging function"""
        if self.p.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """Notification of order status"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed] and self.p.printlog:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')

        elif order.status == order.Canceled:
            self.log('Order Canceled !!!')
        elif order.status == order.Margin:
            self.log('Order Margin !!!')
        elif order.status == order.Rejected:
            self.log('Order Rejected !!!')

        self.order = None

    # def notify_trade(self, trade):
    #     """Notification of trade closed"""
    #     if not trade.isclosed:
    #         return
    #     # self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    #     pass
    
    def notify_data(self, data, status, *args, **kwargs):
        """Notification of data status changes (e.g. new bar, end of data)"""
        # self.log(f'Data {data._name} status = {data._getstatusname()}')
        self.stocks[data._name] = status

    def can_trade_now(self, dt):
        """Check if current time is within trading hours"""
        current_time = dt.time()
        # Check day of week
        weekday = dt.weekday()
        if weekday not in self.p.trade_weekdays:
            return False

        # Check trading hours
        if self.p.trade_start_tm <= current_time <= self.p.trade_end_tm:
            return True
        return False

    def get_position_size(self, data):
        """Calculate position size based on portfolio percentage"""
        # Get available cash
        cash = self.broker.getvalue() # getcash()
        # Calculate position value
        position_value = cash * (self.p.volume_pct / 100.0)
        # Get current price
        price = data.close[0]
        if price == 0:
            return 0
        # Calculate size
        size = position_value / price
        # Round to reasonable decimals
        size = round(size)
        return size

    def count_open_positions(self):
        """Count number of currently open positions"""
        count = 0
        for d in self.datas:
            if self.getposition(d).size > 0:
                count += 1
        return count

    def next(self):
        """Main strategy logic - called for each bar"""

        # Process each data feed
        for d in self.datas:
            dt = d.datetime.datetime(0)

            if (self.last_time_set_clusters is None or
                self.last_time_set_clusters != dt):

                if self.p.volatility_cluster != 0:
                    try:
                        self.volatility_clusters.calculate(self.datas)
                        self.last_time_set_clusters = dt

                        # Optional: Print cluster info
                        if self.p.printlog:
                            info = self.volatility_clusters.get_cluster_info()
                            # self.log(f"Clusters updated ---- C1: {info['cluster_one']['count']}, {info['cluster_one']['names']}")
                            # self.log(f"Clusters updated - C1: {info['cluster_one']['count']}, "
                            #        f"C2: {info['cluster_two']['count']}, "
                            #        f"C3: {info['cluster_three']['count']}")
                    except Exception as e:
                        if self.p.printlog:
                            self.log(f"Error calculating clusters: {e}")

            # Check if we can trade at this time
            if not self.can_trade_now(dt):
                continue

            # Check volatility cluster filter
            # Only apply filter if volatility_cluster parameter is set (1-3)
            # and we have no position in this instrument
            position = self.getposition(d)
            if position.size == 0 and self.p.volatility_cluster != 0:
                # Check if this instrument is in the selected cluster
                if not self.volatility_clusters.is_in_cluster(d, self.p.volatility_cluster):
                    continue  # Skip this instrument if not in selected cluster

            # Get indicators
            lr_channel = self.inds[d._name]['lr_channel']

            # Check if indicators are ready
            if np.isnan(lr_channel.upperband[0]) or np.isnan(lr_channel.lowerband[0]):
                continue

            # Get current close price
            close = d.close[0]

            if self.p.live_mode and self.stocks[d._name] != bt.DataBase.LIVE:
                continue  # Skip if not live in live mode

            # --- ENTRY LOGIC ---
            if position.size == 0:
                # Check max positions limit
                if self.count_open_positions() >= self.p.max_positions:
                    continue

                # Check if price closed above upper band
                if close >= lr_channel.upperband[0]:

                    # Apply SMA filter if enabled
                    if self.p.sma_filter:
                        sma = self.inds[d._name]['sma']
                        if close < sma[0]:
                            continue

                    # Calculate position size
                    size = self.get_position_size(d)

                    if size > 0:
                        # Enter long position
                        # self.log(f'BUY CREATE {d._name}, Price: {close:.2f}, Size: {size:.2f}')
                        self.buy(data=d, size=size)

            # --- EXIT LOGIC ---
            elif position.size > 0:
                # Check if price closed below lower band
                if close <= lr_channel.lowerband[0]:
                    # Close position
                    # self.log(f'SELL CREATE {d._name}, Price: {close:.2f}, Size: {position.size:.2f}')
                    self.close(data=d)

    #def stop(self):
    #    """Called when backtesting is finished"""
    #    self.log(f'Ending Value: {self.broker.getvalue():.2f}, ', doprint=True)


