# Три Солдата на скринерах c фильтрацией по группам волатильности
Пример торговой стратегии на идее из **OS Engine AlgoStart2Soldiers**

### Установка и настройка

#### Способ 1: Установка python менеджера UV через встроенный скрипт
```bash
# На Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# На Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Способ 2: Установка UV через pip
```bash
pip install uv
```

#### Инициализация проекта
```bash
uv sync
```

### Файлы проекта
- `R2_Soldiers.py` - торговая стратегия
- `indicators.py` - доп индикаторы используемые в стратегии
- `R_common.py` - общие параметры и список торгуемых тикеров акций
- `R2_test.py` - запуск стратегии на тестовых данных

### Готовые данные 30мин свечек для тестирования
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


### Запуск примеров
**В примерах у стратегии УЖЕ введены оптимальные параметры**

`R2_test.py`  - запуск стратегии на тестовых данных из папки DATA
```bash
uv run R2_test.py
```

Как реализовать Live торговлю смотри в первом примере **R1_LinearRegression**


### Статистика результатов торговой стратегии

```
Strategy                                R2_Soldiers
sma_filter                                     True
sma_period                                      150
volume_pct                                       10
max_positions                                    10
volatility_cluster                                3
cluster_lookback                                 80
days_volatility_adaptive                          7
height_soldiers_vola_percent                     80
proc_height_take                                185
proc_height_stop                                106
trade_start_tm                             10:05:00
trade_end_tm                               18:00:00
trade_weekdays                      [0, 1, 2, 3, 4]
iceberg_count                                     1
printlog                                      False
live_mode                                     False
Start                           2015-01-05 10:00:00
End                             2025-12-24 11:00:00
Duration                         4006 days 01:00:00
Equity Start [$]                          1000000.0
Equity Final [$]                       3866304.5288
Equity Peak [$]                        3913830.6743
Commissions [$]                         270239.7199
Cum Return [%]                             286.6305
Return (Ann.) [%]                           12.8449
Volatility (Ann.) [%]                        8.5355
CAGR [%]                                       8.88
Sharpe Ratio                                 1.4586
Skew                                         0.9045
Kurtosis                                     8.6912
Smart Sharpe Ratio                           0.6556
Sortino Ratio                                2.3886
VWR Ratio                                     4.652
Calmar Ratio                                 0.9691
Recovery factor [%]                         10.5113
Max. Drawdown [%]                          -13.2546
Avg. Drawdown [%]                           -0.5392
Max. Drawdown Duration            457 days 22:00:00
Avg. Drawdown Duration              6 days 08:04:00
Drawdown Peak                   2015-10-02 15:30:00
# Trades                                       1589
Win Rate [%]                                46.5072
Best Trade [%]                              37.5191
Worst Trade [%]                            -20.6805
Avg. Trade [%]                               0.8283
Max. Trade Duration               430 days 17:00:00
Avg. Trade Duration                 8 days 17:21:00
Profit Factor                                 1.332
Expectancy [%]                               0.0494
SQN                                          6.1532
Kelly Criterion [%]                         15.1723
```

- [output_stats.html](https://alex-shur.github.io/RoboBuilding/R2_Soldiers/output_stats.html) - quantstats like strategy report

![chart1](https://raw.githubusercontent.com/Alex-Shur/RoboBuilding/master/Stocks/R2_Soldiers/scr1.png)
