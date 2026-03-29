# Стратегия "Адаптивный Ценовой Канал" с фильтром по волатильности
Пример торговой стратегии на идее из **OS Engine AlgoStart3PriceChannel**

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
- `R3_PriceChannel.py` - торговая стратегия
- `indicators.py` - доп индикаторы используемые в стратегии
- `R_common.py` - общие параметры и список торгуемых тикеров акций
- `R3_test.py` - запуск стратегии на тестовых данных

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

`R3_test.py`  - запуск стратегии на тестовых данных из папки DATA
```bash
uv run R3_test.py
```

Как реализовать Live торговлю смотри в первом примере **R1_LinearRegression**


### Статистика результатов торговой стратегии

```
Strategy                          R3_PriceChannel
pc_adx_length                                  50
pc_ratio                                      840
sma_filter                                   True
sma_period                                     70
volume_pct                                     10
max_positions                                  10
volatility_cluster                              2
cluster_lookback                              100
trade_start_tm                           10:05:00
trade_end_tm                             18:00:00
trade_weekdays                    [0, 1, 2, 3, 4]
iceberg_count                                   1
printlog                                    False
live_mode                                   False
Start                         2015-01-05 10:00:00
End                           2025-12-24 11:00:00
Duration                       4006 days 01:00:00
Equity Start [$]                        1000000.0
Equity Final [$]                     6495894.3642
Equity Peak [$]                      7705133.7873
Commissions [$]                        783007.139
Cum Return [%]                           549.5894
Return (Ann.) [%]                         18.2004
Volatility (Ann.) [%]                     13.6091
CAGR [%]                                  12.4915
Sharpe Ratio                               1.2968
Skew                                       0.7433
Kurtosis                                   6.1401
Smart Sharpe Ratio                         0.9472
Sortino Ratio                              2.0901
VWR Ratio                                  6.3095
Calmar Ratio                               0.9715
Recovery factor [%]                       10.5412
Max. Drawdown [%]                        -18.7351
Avg. Drawdown [%]                         -0.8213
Max. Drawdown Duration          392 days 06:00:00
Max. Drawdown Duration [D]                 392.25
Avg. Drawdown Duration            5 days 22:25:00
Avg. Drawdown Duration [D]                 5.9334
Drawdown Peak                 2025-10-16 11:30:00
# Trades                                     2776
Win Rate [%]                              44.7406
Best Trade [%]                            86.2106
Worst Trade [%]                           -16.613
Avg. Trade [%]                             0.6765
Max. Trade Duration              79 days 21:30:00
Avg. Trade Duration              10 days 03:42:00
Profit Factor                              1.2617
Expectancy [%]                               0.07
SQN                                        5.5862
Kelly Criterion [%]                       13.0207
```

- [output_stats.html](https://alex-shur.github.io/RoboBuilding/R3_PriceChannel/output_stats.html) - quantstats like strategy report

![chart1](https://raw.githubusercontent.com/Alex-Shur/RoboBuilding/master/Stocks/R3_PriceChannel/scr1.png)
