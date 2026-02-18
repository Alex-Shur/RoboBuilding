# RoboBuilding
Примеры торговых стратегий - Samples of algotrading strategies

# Готовые данные 30мин свечек для тестирования
- [DATA.zip](https://drive.google.com/file/d/1kzSEoLYyxrRTQBSAUN2Y8u3FhcbQlewH/view?usp=sharing)
- Также данные можно скачать самостоятельно с помощью [MOEX-Downloader](https://github.com/Alex-Shur/moex-downloader)
- Вам нужны 30мин свечки для следующих тикеров:
```
AFKS,  AFLT,  ALRS,  BSPB,  CHMF, 
FEES,  GAZP,  GMKN,  HYDR,  IRAO, 
LKOH,  MAGN,  MGNT,  MOEX,  MTLR, 
MTSS,  NLMK,  NVTK,  PHOR,  PIKK, 
PLZL,  ROSN,  RTKM,  RUAL,  SBERP,
SBER,  SIBN,  SNGSP, SNGS,  TATNP, 
TATN,  TRNFP, UPRO,  VTBR
```
Всего 34 акции, за период с 01.01.2015 по 24.12.2025


```
Strategy                  R1_LinearRegression
lr_period                                 180
lr_deviation                              2.5
sma_filter                               True
sma_period                                286
volume_pct                                 10
max_positions                              10
volatility_cluster                          1
cluster_lookback                           30
trade_start_tm                       10:05:00
trade_end_tm                         18:00:00
trade_weekdays                [0, 1, 2, 3, 4]
iceberg_count                               1
printlog                                False
Start                     2015-01-05 10:00:00
End                       2025-12-24 11:00:00
Duration                   4006 days 01:00:00
Equity Start [$]                    1000000.0
Equity Final [$]                 3378006.1094
Equity Peak [$]                  3434003.6385
Commissions [$]                    215502.029
Cum Return [%]                       237.8006
Return (Ann.) [%]                     11.4916
Volatility (Ann.) [%]                  6.8113
CAGR [%]                                 7.96
Sharpe Ratio                           1.6312
Skew                                    1.999
Kurtosis                              20.5899
Smart Sharpe Ratio                    -1.2059
Sortino Ratio                          2.9402
VWR Ratio                               4.233
Calmar Ratio                           1.1642
Recovery factor [%]                   12.5963
Max. Drawdown [%]                     -9.8707
Avg. Drawdown [%]                     -0.4435
Max. Drawdown Duration      329 days 00:30:00
Avg. Drawdown Duration        6 days 16:38:00
Drawdown Peak             2022-05-26 15:00:00
# Trades                                 1335
Win Rate [%]                          49.2884
Best Trade [%]                        57.9426
Worst Trade [%]                       -18.041
Avg. Trade [%]                         0.9141
Max. Trade Duration          60 days 20:00:00
Avg. Trade Duration           7 days 21:15:00
Profit Factor                           1.444
Expectancy [%]                         0.0441
SQN                                    6.8005
Kelly Criterion [%]                   23.9275
```
- [output_stats.html](https://alex-shur.github.io/RoboBuilding/R1_LinearRegression/output_stats.html) - quantstats like strategy report

![chart1](https://raw.githubusercontent.com/Alex-Shur/RoboBuilding/master/Stocks/R1_LinearRegression/scr1.png)
