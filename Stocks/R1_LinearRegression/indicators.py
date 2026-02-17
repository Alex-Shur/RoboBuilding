"""
MIT License

Copyright (c) 2026 Alex Shurenberg
"""

from shutil import move
import numba
import backtrader_next as bt
import numpy as np

__all__ = ['LinearRegressionChannel_MAD', 'VolatilityStageClusters']


# JIT-compiled Linear Regression with MAD (Mean Absolute Deviation) - C# style
@numba.njit
def compute_linear_regression_mad_numba(data_points, period):
    """
    Calculate linear regression for a window of data using C# algorithm
    Uses Mean Absolute Deviation (MAD) instead of Standard Deviation
    
    This matches the C# LinearRegressionChannelFast_Indicator implementation
    
    Returns: current_regression, mad (mean absolute deviation)
    
    Improvements for higher precision:
    - Uses float64 consistently throughout
    - Kahan summation for numerical stability
    - Pre-calculated sumx and sumx2 to avoid repeated calculations
    """
    # Variables matching C# implementation (explicit float64)
    sumy = np.float64(0.0)
    sumxy = np.float64(0.0)
    
    # Pre-calculate sumx and sumx2 for indices 0..(period-1)
    # sumx = 0 + 1 + 2 + ... + (period-1) = period * (period - 1) / 2
    # sumx2 = 0^2 + 1^2 + 2^2 + ... + (period-1)^2 = (period-1) * period * (2*period-1) / 6
    sumx = np.float64(period * (period - 1)) / 2.0
    sumx2 = np.float64((period - 1) * period * (2 * period - 1)) / 6.0
    
    # Calculate sums with Kahan summation for better numerical stability
    sumy_c = 0.0  # Compensation for lost low-order bits
    sumxy_c = 0.0
    
    for g in range(period):
        y_val = np.float64(data_points[g])
        
        # Kahan summation for sumy
        y_temp = y_val - sumy_c
        t = sumy + y_temp
        sumy_c = (t - sumy) - y_temp
        sumy = t
        
        # Kahan summation for sumxy
        xy_val = y_val * np.float64(g)
        xy_temp = xy_val - sumxy_c
        t = sumxy + xy_temp
        sumxy_c = (t - sumxy) - xy_temp
        sumxy = t
    
    # Calculate c (denominator) with explicit float64
    c = sumx2 * np.float64(period) - sumx * sumx
    
    if abs(c) < 1e-10:  # More robust zero check
        return np.float64(0.0), np.float64(0.0)
    
    # Line equation coefficients (matching C# algorithm)
    b = (sumxy * np.float64(period) - sumx * sumy) / c  # slope
    a = (sumy - sumx * b) / np.float64(period)           # intercept
    
    # Calculate MAD using Kahan summation for numerical stability
    mad = np.float64(0.0)
    mad_c = 0.0  # Compensation
    
    for i in range(period):
        # Calculate regression value at point i
        regression_val = a + b * np.float64(i)
        
        # Absolute distance between point and regression line
        distance = abs(np.float64(data_points[i]) - regression_val)
        
        # Kahan summation for MAD
        temp = distance - mad_c
        t = mad + temp
        mad_c = (t - mad) - temp
        mad = t
    
    mad = mad / np.float64(period)
    
    # Current regression value (last point in window)
    current_regression = a + b * np.float64(period - 1)
    
    return current_regression, mad


@numba.njit
def compute_lr_channel_mad_numba(closes, period, up_deviation, down_deviation):
    """
    Compute Linear Regression Channel using MAD (Mean Absolute Deviation)
    Matches C# LinearRegressionChannelFast_Indicator implementation
    """
    n = len(closes)
    upperband = np.empty(n, dtype=np.float64)
    regression = np.empty(n, dtype=np.float64)
    lowerband = np.empty(n, dtype=np.float64)
    
    # Fill initial values with NaN
    upperband[:period-1] = np.nan
    regression[:period-1] = np.nan
    lowerband[:period-1] = np.nan
    
    # Calculate for each bar starting from period
    for i in range(period-1, n):
        data_window = closes[i-period+1:i+1]
        current_regression, mad = compute_linear_regression_mad_numba(data_window, period)
        
        regression[i] = current_regression
        upperband[i] = current_regression + (mad * up_deviation)
        lowerband[i] = current_regression - (mad * down_deviation)
    
    return upperband, regression, lowerband


class LinearRegressionChannel_MAD(bt.Indicator):
    """
    Linear Regression Channel Indicator with MAD (Mean Absolute Deviation)
    
    This implementation matches the C# LinearRegressionChannelFast_Indicator
    Uses Mean Absolute Deviation instead of Standard Deviation for band calculation
    
    Key differences from LinearRegressionChannel:
    - Uses MAD (Mean Absolute Deviation) instead of StdDev
    - Separate up and down deviation multipliers
    - Default parameters match C# version
    """
    lines = ('upperband', 'regression', 'lowerband')
    params = (
        ('period', 100),          # Length in C#
        ('up_deviation', 2.0),    # Up channel deviation in C#
        ('down_deviation', 2.0),  # Down channel deviation in C#
    )
    plotinfo = dict(subplot=False)
    plotlines = dict(
        regression=dict(_plotskip=True),  # Don't plot regression line
    )

    def __init__(self):
        self.addminperiod(self.p.period)

    def once(self, start, end):
        """Batch processing using numba (vectorized)"""
        if end-start==1:
            return
        
        closes = np.asarray(self.data.get_array_preloaded(), dtype=np.float64)
        
        upperband, regression, lowerband = compute_lr_channel_mad_numba(
            closes,
            self.p.period,
            self.p.up_deviation,
            self.p.down_deviation
        )
        
        self.lines.upperband.ndbuffer(upperband)
        self.lines.regression.ndbuffer(regression)
        self.lines.lowerband.ndbuffer(lowerband)

    def next(self):
        """Incremental calculation for live/next bar updates"""
        # Get data for regression
        data_points = np.array(self.data.get(size=self.p.period), dtype=np.float64)
        
        if len(data_points) < self.p.period:
            self.lines.upperband[0] = float('nan')
            self.lines.regression[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            return
        
        # Calculate using numba function
        current_regression, mad = compute_linear_regression_mad_numba(data_points, self.p.period)
        
        # Calculate bands using MAD
        self.lines.upperband[0] = current_regression + (mad * self.p.up_deviation)
        self.lines.regression[0] = current_regression
        self.lines.lowerband[0] = current_regression - (mad * self.p.down_deviation)



class VolatilityCalculator:
    """
    Volatility Calculator for a single instrument (Numba optimized)
    Calculates volatility as percentage move over period
    
    Matches C# SourceVolatility.Calculate() implementation:
    - Iterates backwards from last candle
    - Takes up to 'period' candles
    - Finds max(High) and min(Low) over period
    - Returns: (max - min) / (min / 100) = percentage move
    """
    
    def __init__(self, data, period):
        self.data = data
        self.period = period
        self.volatility = 0.0
    
    def calculate(self):
        """Calculate volatility as percentage range over period (Numba optimized)"""
        if len(self.data) < 1:
            return 0.0
        
        # Extract high and low arrays - matching C# logic
        # C#: for (int i = Candles.Count - 1; i >= 0 && i > Candles.Count -1-candlesCount; i--)
        # This means: start from last candle, go back up to 'period' candles
        try:
            data_len = len(self.data)
            actual_period = min(self.period, data_len)
            
            # Optimized: get data using slicing instead of loop with negative indexing
            # Take last 'actual_period' elements using negative slicing
            highs = np.array(self.data.high.get(ago=0, size=actual_period), dtype=np.float64)
            lows = np.array(self.data.low.get(ago=0, size=actual_period), dtype=np.float64)
            
            if len(highs) == 0 or len(lows) == 0:
                self.volatility = 0.0
                return self.volatility
            
            # Calculate volatility as percentage range over period
            max_price = np.max(highs)
            min_price = np.min(lows)
    
            if min_price == 0.0:
                self.volatility = 0.0
                return self.volatility
    
            move = max_price - min_price
            self.volatility = move / (min_price / 100.0)
            return self.volatility
        except Exception:
            self.volatility = 0.0
            return self.volatility


class VolatilityStageClusters:
    """
    Volatility Stage Clusters
    
    Divides multiple instruments into 3 clusters based on volatility:
    - ClusterOne: Lowest volatility (33% by default)
    - ClusterTwo: Medium volatility (33% by default)
    - ClusterThree: Highest volatility (34% by default)
    
    This is a direct port from OSEngine VolatilityStageClusters.cs
    """
    
    def __init__(self, lookback=100, one_percent=33.0, two_percent=33.0, three_percent=34.0):
        self.cluster_one = {}
        self.cluster_two = {}
        self.cluster_three = {}
        self.cluster_one_full = []
        self.cluster_two_full = []
        self.cluster_three_full = []
        self.one_lot = 0.0

        self.length = lookback
        self.cluster_one_percent = one_percent
        self.cluster_two_percent = two_percent
        self.cluster_three_percent = three_percent
        # Validate percentages
        total_percent = one_percent + two_percent + three_percent
        if abs(total_percent - 100.0) > 0.01:  # Allow small floating point error
            raise ValueError(f"VolatilityStageClusters error. Percent is not 100. Got {total_percent}")


    def calculate(self, data_feeds):
        """
        Calculate volatility clusters for multiple data feeds
        
        Args:
            data_feeds: List of backtrader data feeds
            candles_count: Number of candles to use for volatility calculation
            cluster_one_pct: Percentage for cluster one (lowest volatility)
            cluster_two_pct: Percentage for cluster two (medium volatility)
            cluster_three_pct: Percentage for cluster three (highest volatility)
        """
        # Clear previous clusters
        self.cluster_one.clear()
        self.cluster_two.clear()
        self.cluster_three.clear()
        self.cluster_one_full.clear()
        self.cluster_two_full.clear()
        self.cluster_three_full.clear()
        self._calculate_clusters(data_feeds)
    
    def _calculate_clusters(self, data_feeds):
        """Internal method to calculate and assign clusters (Numba optimized)"""
        if not data_feeds:
            return
        
        # Calculate volatility for each data feed
        sources_with_volatility = []
        
        for data in data_feeds:
            # Check if data has enough candles
            if len(data) < self.length:
                continue
            
            # Calculate volatility
            vol_calc = VolatilityCalculator(data, self.length)
            volatility = vol_calc.calculate()
            
            sources_with_volatility.append({
                'data': data,
                'volatility': volatility,
                'name': data._name if hasattr(data, '_name') else 'Unknown'
            })
        
        # Need at least 2 instruments to cluster
        if len(sources_with_volatility) <= 1:
            if len(sources_with_volatility) == 1:
                # Put single instrument in cluster one
                self.cluster_one.append(sources_with_volatility[0]['data'])
            return

        # Sort by volatility (ascending - lowest to highest)
        sources_with_volatility.sort(key=lambda x: x['volatility'])

        total_count = len(sources_with_volatility)
        self.one_lot = float(total_count) / 100.0
        cluster_one_limit = self.cluster_one_percent * self.one_lot
        cluster_two_limit = (self.cluster_one_percent + self.cluster_two_percent) * self.one_lot

        for i in range(total_count):
            position = float(i + 1)
            if position <= cluster_one_limit:
                self.cluster_one_full.append(sources_with_volatility[i])
                self.cluster_one[sources_with_volatility[i]['name']] = sources_with_volatility[i]
            elif position <= cluster_two_limit:
                self.cluster_two_full.append(sources_with_volatility[i])
                self.cluster_two[sources_with_volatility[i]['name']] = sources_with_volatility[i]
            else:
                self.cluster_three_full.append(sources_with_volatility[i])
                self.cluster_three[sources_with_volatility[i]['name']] = sources_with_volatility[i]
    
    def is_in_cluster(self, data, cluster_number):
        """
        Check if data feed is in specified cluster
        
        Args:
            data: Backtrader data feed
            cluster_number: 1, 2, or 3
        
        Returns:
            bool: True if data is in specified cluster
        """
        if cluster_number == 1:
            return data._name in self.cluster_one
        elif cluster_number == 2:
            return data._name in self.cluster_two
        elif cluster_number == 3:
            return data._name in self.cluster_three
        return False
    
    def get_cluster_info(self):
        """Get information about current clusters"""
        return {
            'cluster_one': {
                'count': len(self.cluster_one),
                'names': [f"{v['name']}={v['volatility']:.4f}" for v in self.cluster_one_full],
                'one_lot': self.one_lot
            },
            'cluster_two': {
                'count': len(self.cluster_two),
                'names': [f"{v['name']}={v['volatility']:.4f}" for v in self.cluster_two_full],
                'one_lot': self.one_lot
            },
            'cluster_three': {
                'count': len(self.cluster_three),
                'names': [f"{v['name']}={v['volatility']:.4f}" for v in self.cluster_three_full],
                'one_lot': self.one_lot
            }
        }


