"""
MIT License

Copyright (c) 2026 Alex Shurenberg

== Three Soldiers Pattern with Volatility Adaptation ==

Based on OSEngine C# Robot: AlgoStart2Soldiers

Strategy Description:
- Trend following strategy based on three consecutive growing candles pattern
- Uses volatility adaptation to adjust candle size requirements
- Entry: Three growing candles of certain size relative to volatility
- Exit: Stop and Profit based on pattern height
- Optional SMA filter to confirm trend direction
- Uses volatility clustering for instrument selection

Original Author: AlexWan (OsEngine)
"""

from collections import defaultdict
import datetime
import pandas as pd
import backtrader_next as bt
from backtrader_next.feeds.pandafeed import PandasData

import numpy as np
from datetime import time
from numba import njit

from indicators import VolatilityStageClusters
from R_common import stock_names


@njit(cache=True)
def _calc_daily_volatility(highs: np.ndarray, lows: np.ndarray,
                           date_ints: np.ndarray, days_needed: int) -> np.ndarray:
    """
    Numba-compiled inner loop for adapt_soldiers_height.

    Parameters
    ----------
    highs, lows  : float64 arrays, index 0 = current bar, going backwards in time
    date_ints    : int32 array, int(matplotlib_date) — unique integer per calendar day
    days_needed  : how many complete days of volatility to collect

    Returns
    -------
    1-D float64 array of per-day volatility percentages (length <= days_needed)
    """
    vola = np.empty(days_needed, dtype=np.float64)
    count = 0

    min_val = np.inf
    max_val = -np.inf
    current_day = date_ints[0]

    n = len(highs)
    for i in range(n):
        day = date_ints[i]

        if day != current_day:          # day boundary — save completed day
            if max_val > -np.inf and min_val < np.inf and min_val > 0.0:
                vola[count] = (max_val - min_val) / min_val * 100.0
                count += 1
                if count >= days_needed:
                    break
            current_day = day
            min_val = np.inf
            max_val = -np.inf

        h = highs[i]
        l = lows[i]
        if h > max_val:
            max_val = h
        if l < min_val:
            min_val = l

    # last (possibly incomplete) day
    if count < days_needed and max_val > -np.inf and min_val > 0.0:
        vola[count] = (max_val - min_val) / min_val * 100.0
        count += 1

    return vola[:count]


class R2_Soldiers(bt.Strategy):
    """
    Three Soldiers Pattern Strategy with Volatility Adaptation
    
    The strategy looks for three consecutive growing candles (bullish pattern)
    where the total height meets volatility-based criteria.
    
    Parameters:
    - sma_filter: Enable SMA filter (default: True)
    - sma_period: SMA filter period (default: 150)
    - volume_pct: Position size as % of portfolio (default: 10)
    - max_positions: Maximum concurrent positions (default: 10)
    - volatility_cluster: Trade only this volatility cluster 1-3 (default: 3)
    - cluster_lookback: Period for volatility clustering (default: 80)
    - days_volatility_adaptive: Days to calculate volatility (default: 7)
    - height_soldiers_vola_percent: Required pattern height as % of daily volatility (default: 80)
    - proc_height_take: Profit target as % of pattern height (default: 185)
    - proc_height_stop: Stop loss as % of pattern height (default: 106)
    """
    
    params = dict(
        # SMA Filter settings
        sma_filter=True,
        sma_period=150,   # 10 - 300 #10
        
        # Position sizing
        volume_pct=10,    # 1 - 50  #4
        max_positions=10, # 0 - 20  #1
        
        # Volatility clustering
        volatility_cluster=3,  # 1 - 3
        cluster_lookback=80,   # 10 - 300  #1
        
        # Volatility adaptation
        days_volatility_adaptive=7,  # 0 - 20  #1
        height_soldiers_vola_percent=80,  # 0 - 20 #1

        # Stop and Take parameters
        proc_height_take=185,  # 0 - 20 #1
        proc_height_stop=106,  # 0 - 20 #1
        
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
        self.s_datas = list() # strategy datas
        
        # Volatility clusters manager
        self.volatility_clusters = VolatilityStageClusters(
            lookback=self.p.cluster_lookback,
            one_percent=33.0,    # Lowest volatility
            two_percent=33.0,    # Medium volatility
            three_percent=34.0   # Highest volatility
        )
        self.last_time_set_clusters = None
        
        # Dictionary to store indicators for each data feed
        self.inds = {}
        # Dictionary to store trade settings for each instrument
        # Includes adaptive volatility calculations
        self.trade_settings = {}

        self.stocks = defaultdict(lambda: bt.DataBase.UNKNOWN)
        for name in stock_names:
            d = self.getdatabyname(name)
            if d is None:
                continue
            self.stocks[name] = bt.DataBase.UNKNOWN
            self.s_datas.append(d)

            # Initialize trade settings for this instrument
            self.trade_settings[name] = {
                'height_soldiers': 0,  # Adaptive pattern height requirement
                'vola_pct_sma': 0,     # Average daily volatility % (used by both modes)
                'last_update_date': None,  # Last date when volatility was calculated
            }

            self.inds[name] = {}
            # SMA Filter (optional)
            if self.p.sma_filter:
                self.inds[name]['sma'] = bt.indicators.SMA(
                    d.close,
                    period=self.p.sma_period
                )


    def log(self, txt, dt=None, doprint=False):
        """Logging function"""
        if self.p.printlog or doprint:
            dt = dt or self.s_datas[0].datetime.datetime(0)
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
        cash = self.broker.getvalue()
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
        for d in self.s_datas:
            if self.getposition(d).size > 0:
                count += 1
        return count

    def adapt_soldiers_height(self, data):
        """
        Calculate adaptive pattern height requirement based on recent volatility.

        The inner loop is compiled with Numba (@njit) for maximum performance.
        Backtrader matplotlib-date floats are used directly: int(mpl_float)
        gives a unique integer per calendar day, so no Python datetime objects
        are needed inside the hot loop.
        """
        if self.p.days_volatility_adaptive <= 0 or self.p.height_soldiers_vola_percent <= 0:
            return

        settings = self.trade_settings[data._name]
        current_dt = data.datetime.datetime(0)

        # Update once per day
        if settings['last_update_date'] == current_dt.date():
            return

        # Approximate: 20 bars per day for 30-min timeframe
        required_bars = self.p.days_volatility_adaptive * 20

        if len(data) < required_bars:
            return

        # ---- Extract raw arrays from backtrader LineBuffers ----
        # Use line.get() — works in both normal and QBuffer (exactbars) modes.
        # get(size, ago=0) returns a list oldest→newest, so we reverse to get
        # index 0 = current bar (same convention as the Numba function).
        highs     = np.array(data.high.get(size=required_bars, ago=0),
                             dtype=np.float64)[::-1]
        lows      = np.array(data.low.get(size=required_bars, ago=0),
                             dtype=np.float64)[::-1]
        # Matplotlib date floats: integer part = unique calendar day number
        date_ints = np.array(data.datetime.get(size=required_bars, ago=0),
                             dtype=np.float64)[::-1].astype(np.int32)

        # ---- Numba hot loop ----
        vola_days = _calc_daily_volatility(
            highs, lows, date_ints, self.p.days_volatility_adaptive
        )

        if len(vola_days) == 0:
            return

        vola_percent_sma  = vola_days.mean()
        all_soldiers_height = vola_percent_sma * (self.p.height_soldiers_vola_percent / 100)

        settings['height_soldiers']   = all_soldiers_height
        settings['last_update_date']  = current_dt.date()

        if self.p.printlog:
            self.log(f"{data._name}: Adaptive height updated: {all_soldiers_height:.2f}% "
                     f"(avg daily vola: {vola_percent_sma:.2f}%)")

    def check_three_soldiers_pattern(self, data):
        """
        Check if the last three candles form a valid "three soldiers" pattern.

        Returns: True if pattern is valid, False otherwise.

        Candles: [-1] newest, [-2] middle, [-3] oldest  (same as original).

        Optimisation notes:
        - height_soldiers guard checked first (most likely to fail early).
        - Direct indexed access instead of for-loop over 3 elements.
        - No abs() — bullish pattern guarantees close[-1] > open[-3].
        """
        settings = self.trade_settings[data._name]

        # Cheapest guard first — skip if adaptive threshold not yet calculated
        height_soldiers = settings['height_soldiers']
        if height_soldiers == 0:
            return False

        # Need at least 3 candles
        if len(data) < 3:
            return False

        # Cache values — avoids repeated LineBuffer __getitem__ calls
        c1 = data.close[0];  o1 = data.open[0]    # newest candle
        c2 = data.close[-1]; o2 = data.open[-1]   # middle candle
        c3 = data.close[-2]; o3 = data.open[-2]   # oldest candle

        # All three candles must be bullish (close > open)
        if c1 <= o1 or c2 <= o2 or c3 <= o3:
            return False

        # Pattern height: open of oldest → close of newest (always positive for bullish)
        if c1 == 0:
            return False

        pattern_height_percent = (c1 - o3) / c1 * 100

        return pattern_height_percent >= height_soldiers

    def next(self):
        """Main strategy logic - called for each bar"""
        
        # Process each data feed
        for d in self.s_datas:
            dt = d.datetime.datetime(0)
            
            if (self.last_time_set_clusters is None or
                self.last_time_set_clusters != dt):

                if self.p.volatility_cluster != 0:
                    try:
                        self.volatility_clusters.calculate(self.s_datas)
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
            
            # Update adaptive soldiers height
            self.adapt_soldiers_height(d)
            
            # Check volatility cluster filter
            # Only apply filter if volatility_cluster parameter is set (1-3)
            # and we have no position in this instrument
            position = self.getposition(d)
            
            if position.size == 0 and self.p.volatility_cluster != 0:
                # Check if this instrument is in the selected cluster
                if not self.volatility_clusters.is_in_cluster(d, self.p.volatility_cluster):
                    continue  # Skip this instrument if not in selected cluster
            
            # Get current close price
            close = d.close[0]
            
            if self.p.live_mode and self.stocks[d._name] != bt.DataBase.LIVE:
                continue  # Skip if not live in live mode

            # --- ENTRY LOGIC ---
            if position.size == 0:
                # Check max positions limit
                if self.count_open_positions() >= self.p.max_positions:
                    continue
                
                # Check for three soldiers pattern
                if self.check_three_soldiers_pattern(d):
                    
                    # Apply SMA filter if enabled
                    if self.p.sma_filter:
                        sma = self.inds[d._name]['sma']
                        
                        if len(sma) < 2:
                            continue
                        
                        # SMA should be rising
                        if sma[0] < sma[-1]:
                            continue
                    
                    # Calculate position size
                    size = self.get_position_size(d)
                    
                    if size > 0:
                        # Enter long position
                        if self.p.printlog:
                            self.log(f'BUY CREATE {d._name}, Price: {close:.2f}, Size: {size:.2f}')
                        
                        order = self.buy(data=d, size=size)
                        
                        # Store entry information for exit calculations
                        if order:
                            # Calculate pattern height for stop/take calculation
                            pattern_height = abs(d.open[-2] - d.close[0])
                            
                            # Calculate stop and take prices
                            last_price = d.close[0]
                            price_stop = last_price - (pattern_height * self.p.proc_height_stop / 100)
                            price_take = last_price + (pattern_height * self.p.proc_height_take / 100)
                            
                            # Store in position metadata (we'll use broker for this)
                            # Note: In backtrader, we can't directly store metadata with position
                            # So we'll store it in a dictionary keyed by data
                            if not hasattr(self, 'position_metadata'):
                                self.position_metadata = {}
                            
                            self.position_metadata[d] = {
                                'stop_price': price_stop,
                                'take_price': price_take,
                                'entry_price': last_price,
                            }
                            
                            if self.p.printlog:
                                self.log(f'{d._name}: Stop={price_stop:.2f}, Take={price_take:.2f}, '
                                        f'Pattern Height={pattern_height:.2f}')
            
            # --- EXIT LOGIC ---
            elif position.size > 0:
                # Get stop and take prices from metadata
                if hasattr(self, 'position_metadata') and d in self.position_metadata:
                    metadata = self.position_metadata[d]
                    stop_price = metadata['stop_price']
                    take_price = metadata['take_price']
                    
                    # Check stop loss
                    if close <= stop_price:
                        if self.p.printlog:
                            self.log(f'STOP LOSS HIT {d._name}, Price: {close:.2f}, Stop: {stop_price:.2f}')
                        self.close(data=d)
                        # Clean up metadata
                        del self.position_metadata[d]
                    
                    # Check take profit
                    elif close >= take_price:
                        if self.p.printlog:
                            self.log(f'TAKE PROFIT HIT {d._name}, Price: {close:.2f}, Take: {take_price:.2f}')
                        self.close(data=d)
                        # Clean up metadata
                        del self.position_metadata[d]

    #def stop(self):
    #    """Called when backtesting is finished"""
    #    self.log(f'Ending Value: {self.broker.getvalue():.2f}', doprint=True)


