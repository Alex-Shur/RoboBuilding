"""
MIT License

Copyright (c) 2026 Alex Shurenberg
"""

from shutil import move
import numba
import backtrader_next as bt
import numpy as np

__all__ = ['LinearRegressionChannel_MAD', 'VolatilityStageClusters', 'PriceChannelAdaptive', 'PriceChannelAdaptiveNumba', 'ADXNumba']


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


class PriceChannelAdaptive(bt.Indicator):
    """
    Adaptive Price Channel Indicator
    
    This indicator creates a price channel (upper and lower bands) that adapts
    based on the ADX (Average Directional Index) indicator.
    
    The channel length is calculated as: max(ratio / ADX, 1)
    - When ADX is high (strong trend), channel is shorter (more responsive)
    - When ADX is low (weak trend), channel is longer (less responsive)
    
    Port from OSEngine PriceChannelAdaptive.cs
    
    Parameters:
    - adx_period: Period for ADX calculation (default: 10)
    - ratio: Ratio used to calculate adaptive length (default: 100)
    """
    
    lines = ('upperband', 'lowerband', 'adaptive_length')
    params = (
        ('adx_period', 10),   # Adx Period in C#
        ('ratio', 100),       # Ratio in C#
    )
    plotinfo = dict(subplot=False)
    plotlines = dict(
        adaptive_length=dict(_plotskip=True),  # Don't plot adaptive length
    )

    def __init__(self):
        # Create ADX indicator
        self.adx = bt.indicators.AverageDirectionalMovementIndex(
            self.data,
            period=self.p.adx_period
        )
        
        # Ensure enough data for ADX calculation
        self.addminperiod(self.p.adx_period * 2)

    def next(self):
        """Calculate adaptive price channel"""
        # Get current ADX value
        adx_value = self.adx.lines.adx[0]
        
        # Check if ADX is valid
        if adx_value == 0 or np.isnan(adx_value):
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            self.lines.adaptive_length[0] = float('nan')
            return
        
        # Calculate adaptive length: max(ratio / ADX, 1)
        adaptive_length = max(int(self.p.ratio / adx_value), 1)
        self.lines.adaptive_length[0] = adaptive_length
        
        # Make sure we have enough data
        if len(self.data) < adaptive_length:
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            return
        
        # Calculate highest high and lowest low over adaptive_length period
        # Using data.get(size=adaptive_length) to get last N bars
        highs = self.data.high.get(size=adaptive_length)
        lows = self.data.low.get(size=adaptive_length)
        
        if len(highs) < adaptive_length or len(lows) < adaptive_length:
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            return
        
        # Upper band = highest high over period
        self.lines.upperband[0] = max(highs)
        
        # Lower band = lowest low over period
        self.lines.lowerband[0] = min(lows)


@numba.njit
def _compute_max_min_numba(highs, lows):
    """Compute max of highs and min of lows over a window"""
    max_high = highs[0]
    min_low = lows[0]
    for i in range(1, len(highs)):
        if highs[i] > max_high:
            max_high = highs[i]
        if lows[i] < min_low:
            min_low = lows[i]
    return max_high, min_low


@numba.njit
def compute_price_channel_adaptive_numba(highs, lows, adx_values, ratio):
    """
    Compute Adaptive Price Channel for all bars at once.
    For each bar: adaptive_length = max(ratio / ADX, 1)
    upperband = max(high) over adaptive_length bars
    lowerband = min(low)  over adaptive_length bars
    """
    n = len(highs)
    upperband = np.empty(n, dtype=np.float64)
    lowerband = np.empty(n, dtype=np.float64)
    adaptive_lengths = np.empty(n, dtype=np.float64)

    for i in range(n):
        adx_val = adx_values[i]

        if adx_val == 0.0 or np.isnan(adx_val):
            upperband[i] = np.nan
            lowerband[i] = np.nan
            adaptive_lengths[i] = np.nan
            continue

        adaptive_length = max(int(ratio / adx_val), 1)
        adaptive_lengths[i] = float(adaptive_length)

        if i < adaptive_length - 1:
            upperband[i] = np.nan
            lowerband[i] = np.nan
            continue

        start = i - adaptive_length + 1
        max_high = highs[start]
        min_low = lows[start]
        for j in range(start + 1, i + 1):
            if highs[j] > max_high:
                max_high = highs[j]
            if lows[j] < min_low:
                min_low = lows[j]

        upperband[i] = max_high
        lowerband[i] = min_low

    return upperband, lowerband, adaptive_lengths


class PriceChannelAdaptiveNumba(bt.Indicator):
    """
    Adaptive Price Channel Indicator (Numba optimized)

    Creates a price channel (upper/lower bands) that adapts based on ADX:
    - adaptive_length = max(ratio / ADX, 1)
    - When ADX is high (strong trend) → shorter channel (more responsive)
    - When ADX is low  (weak trend)   → longer  channel (less responsive)

    Port from OSEngine PriceChannelAdaptive.cs
    Batch processing via numba (once), incremental via next.

    Parameters:
    - adx_period: Period for ADX calculation (default: 10)
    - ratio:      Ratio used to calculate adaptive length (default: 100)
    """

    lines = ('upperband', 'lowerband', 'adaptive_length')
    params = (
        ('adx_period', 10),
        ('ratio', 100),
    )
    plotinfo = dict(subplot=False)
    plotlines = dict(
        adaptive_length=dict(_plotskip=True),
    )

    def __init__(self):
        self.adx = ADXNumba(
            self.data,
            period=self.p.adx_period
        )
        self.addminperiod(self.p.adx_period * 2)

    def once(self, start, end):
        """Batch processing using numba (vectorized)"""
        if end - start == 1:
            return

        highs = np.asarray(self.data.high.get_array_preloaded(), dtype=np.float64)
        lows = np.asarray(self.data.low.get_array_preloaded(), dtype=np.float64)
        adx_values = np.asarray(self.adx.lines.adx.get_array_preloaded(), dtype=np.float64)

        upperband, lowerband, adaptive_lengths = compute_price_channel_adaptive_numba(
            highs, lows, adx_values, self.p.ratio
        )

        self.lines.upperband.ndbuffer(upperband)
        self.lines.lowerband.ndbuffer(lowerband)
        self.lines.adaptive_length.ndbuffer(adaptive_lengths)

    def next(self):
        """Incremental calculation for live/next bar updates"""
        adx_value = self.adx.lines.adx[0]

        if adx_value == 0 or np.isnan(adx_value):
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            self.lines.adaptive_length[0] = float('nan')
            return

        adaptive_length = max(int(self.p.ratio / adx_value), 1)
        self.lines.adaptive_length[0] = float(adaptive_length)

        if len(self.data) < adaptive_length:
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            return

        highs = np.array(self.data.high.get(size=adaptive_length), dtype=np.float64)
        lows = np.array(self.data.low.get(size=adaptive_length), dtype=np.float64)

        if len(highs) < adaptive_length or len(lows) < adaptive_length:
            self.lines.upperband[0] = float('nan')
            self.lines.lowerband[0] = float('nan')
            return

        max_high, min_low = _compute_max_min_numba(highs, lows)
        self.lines.upperband[0] = max_high
        self.lines.lowerband[0] = min_low


@numba.njit
def compute_adx_numba(highs, lows, closes, period):
    """
    Compute ADX (Average Directional Index) using Wilder's Smoothed Moving Average.

    Formula (Wilder, 1978):
      upmove   = high[i] - high[i-1]
      downmove = low[i-1] - low[i]
      +DM = upmove  if upmove > downmove and upmove > 0 else 0
      -DM = downmove if downmove > upmove and downmove > 0 else 0
      TR  = max(high-low, abs(high-prev_close), abs(low-prev_close))
      +DI = 100 * SMMA(+DM, period) / SMMA(TR, period)
      -DI = 100 * SMMA(-DM, period) / SMMA(TR, period)
      DX  = 100 * abs(+DI - -DI) / (+DI + -DI)
      ADX = SMMA(DX, period)

    Wilder's Smoothed Moving Average (SMMA):
      First value = simple average of first 'period' values
      Next values = (prev_smma * (period - 1) + new_value) / period

    Returns:
      adx_values: array of ADX values (NaN until enough data)
    """
    n = len(highs)
    adx_values = np.empty(n, dtype=np.float64)
    adx_values[:] = np.nan

    if n < period * 2:
        return adx_values

    # --- Step 1: compute raw TR, +DM, -DM for each bar (starting from bar 1) ---
    tr_arr = np.empty(n - 1, dtype=np.float64)
    pdm_arr = np.empty(n - 1, dtype=np.float64)
    mdm_arr = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        upmove = highs[i] - highs[i - 1]
        downmove = lows[i - 1] - lows[i]

        # True Range
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr_arr[i - 1] = max(hl, hpc, lpc)

        # +DM / -DM
        if upmove > downmove and upmove > 0.0:
            pdm_arr[i - 1] = upmove
        else:
            pdm_arr[i - 1] = 0.0

        if downmove > upmove and downmove > 0.0:
            mdm_arr[i - 1] = downmove
        else:
            mdm_arr[i - 1] = 0.0

    # --- Step 2: Wilder SMMA of TR, +DM, -DM ---
    # First value = sum of first 'period' raw values
    smma_tr = np.float64(0.0)
    smma_pdm = np.float64(0.0)
    smma_mdm = np.float64(0.0)
    for i in range(period):
        smma_tr += tr_arr[i]
        smma_pdm += pdm_arr[i]
        smma_mdm += mdm_arr[i]

    # --- Step 3: compute DX for each bar, then SMMA of DX ---
    dx_arr = np.empty(n - 1, dtype=np.float64)
    dx_arr[:period - 1] = np.nan

    # First DX at index (period-1) of dx_arr
    if smma_tr == 0.0:
        dx_arr[period - 1] = 0.0
    else:
        di_plus = 100.0 * smma_pdm / smma_tr
        di_minus = 100.0 * smma_mdm / smma_tr
        denom = di_plus + di_minus
        dx_arr[period - 1] = 0.0 if denom == 0.0 else 100.0 * abs(di_plus - di_minus) / denom

    m = len(tr_arr)  # = n - 1
    for i in range(period, m):
        smma_tr = (smma_tr * (period - 1) + tr_arr[i]) / period
        smma_pdm = (smma_pdm * (period - 1) + pdm_arr[i]) / period
        smma_mdm = (smma_mdm * (period - 1) + mdm_arr[i]) / period
        if smma_tr == 0.0:
            dx_arr[i] = 0.0
        else:
            di_plus = 100.0 * smma_pdm / smma_tr
            di_minus = 100.0 * smma_mdm / smma_tr
            denom = di_plus + di_minus
            dx_arr[i] = 0.0 if denom == 0.0 else 100.0 * abs(di_plus - di_minus) / denom

    # --- Step 4: SMMA of DX to get ADX ---
    # First ADX = simple average of first 'period' valid DX values
    # valid DX starts at index (period-1) in dx_arr, i.e. bar (period) in original
    adx_smma = np.float64(0.0)
    for i in range(period - 1, period - 1 + period):
        adx_smma += dx_arr[i]
    adx_smma /= np.float64(period)

    # bar index in original array: period*2 - 1  (0-based)
    adx_values[period * 2 - 1] = adx_smma

    for i in range(period * 2, n):
        adx_smma = (adx_smma * (period - 1) + dx_arr[i - 1]) / period
        adx_values[i] = adx_smma

    return adx_values


class ADXNumba(bt.Indicator):
    """
    ADX (Average Directional Index) — Numba optimized.

    Direct port of Wilder's original algorithm (1978):
      - True Range (TR)
      - +DM / -DM
      - Smoothed (Wilder) Moving Average for TR, +DM, -DM
      - DX = 100 * |+DI - -DI| / (+DI + -DI)
      - ADX = SMMA(DX, period)

    Matches backtrader AverageDirectionalMovementIndex (movav=Smoothed).
    Exposes a single line 'adx' — drop-in replacement for
    bt.indicators.AverageDirectionalMovementIndex(...).lines.adx

    Parameters:
      period: ADX period (default: 14, Wilder classic)
    """

    lines = ('adx',)
    params = (('period', 14),)
    plotinfo = dict(subplot=True)
    plotlines = dict(adx=dict(_name='ADX'))

    def __init__(self):
        self.addminperiod(self.p.period * 7)

    def once(self, start, end):
        """Batch processing using numba (vectorized)"""
        if end - start == 1:
            return

        highs = np.asarray(self.data.high.get_array_preloaded(), dtype=np.float64)
        lows = np.asarray(self.data.low.get_array_preloaded(), dtype=np.float64)
        closes = np.asarray(self.data.close.get_array_preloaded(), dtype=np.float64)

        adx_values = compute_adx_numba(highs, lows, closes, self.p.period)

        self.lines.adx.ndbuffer(adx_values)

    def next(self):
        """Incremental calculation — recomputes over available window"""
        needed = self.p.period * 7
        if len(self.data) < needed:
            self.lines.adx[0] = float('nan')
            return

        size = len(self.data)
        highs = np.array(self.data.high.get(size=size), dtype=np.float64)
        lows = np.array(self.data.low.get(size=size), dtype=np.float64)
        closes = np.array(self.data.close.get(size=size), dtype=np.float64)

        adx_values = compute_adx_numba(highs, lows, closes, self.p.period)
        self.lines.adx[0] = adx_values[-1]


