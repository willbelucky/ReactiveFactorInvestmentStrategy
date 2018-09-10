# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 9. 10.
"""
from ksif import Portfolio
from ksif.core.columns import *
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from pandas import DataFrame

std = 'std_'
rolling = '12_rolling_'
predicted = 'pred_'

ACTIVE_RISK = 'active_risk'
ACTIVE_RETURN = 'active_return'
INFORMATION_RATIO = 'information_ratio'
TRADING_CAPITAL = 'trading_capital'

VALUE = 'value'
MOMENTUM = 'momentum'
QUALITY = 'quality'
VOLATILITY = 'volatility'

STD_RET_1 = std + RET_1
STD_VALUE = std + VALUE
STD_MOMENTUM = std + MOMENTUM
STD_QUALITY = std + QUALITY
STD_VOLATILITY = std + VOLATILITY

ROLLING_RET_1 = rolling + RET_1
ROLLING_VALUE = rolling + STD_VALUE
ROLLING_MOMENTUM = rolling + STD_MOMENTUM
ROLLING_QUALITY = rolling + STD_QUALITY
ROLLING_VOLATILITY = rolling + STD_VOLATILITY

PRED_VALUE = predicted + VALUE
PRED_MOMENTUM = predicted + MOMENTUM
PRED_QUALITY = predicted + QUALITY
PRED_VOLATILITY = predicted + VOLATILITY

pf = Portfolio()

pf[TRADING_CAPITAL] = pf[TRADING_VOLUME_RATIO] * pf[MKTCAP]

pf = pf.periodic_standardize(factor=RET_1)

pf = pf.periodic_standardize(factor=B_P)
pf = pf.periodic_standardize(factor=E_P)
pf[STD_VALUE] = (pf[std + B_P] + pf[std + E_P]) / 2

pf = pf.periodic_standardize(factor=MOM12_1)
pf[STD_MOMENTUM] = pf[std + MOM12_1]

pf = pf.periodic_standardize(factor=GP_A)
pf[STD_QUALITY] = pf[std + GP_A]

pf = pf.periodic_standardize(factor=VOL_3M)
pf[STD_VOLATILITY] = pf[std + VOL_3M]
# %% Low 25% portfolio
universe = pf.loc[(pf[MKTCAP] > 50000000000) &  # 시총 500억 원 이상
                  (pf[TRADING_CAPITAL] >= 1000000000), :]  # 월 거래액 10억 원 이상
low_quarter = universe.periodic_percentage(min_percentage=0.0, max_percentage=0.25, factor=RET_1, bottom=True)
low_quarter = low_quarter.groupby(by=DATE)[STD_VALUE, STD_MOMENTUM, STD_QUALITY, STD_VOLATILITY].mean()

plt.figure()
ax = plt.axes()
ax.plot(low_quarter[STD_VALUE], color='salmon')
ax.plot(low_quarter[STD_MOMENTUM], color='sandybrown')
ax.plot(low_quarter[STD_QUALITY], color='lightgreen')
ax.plot(low_quarter[STD_VOLATILITY], color='lightblue')
plt.title('Low 25% RET_1')
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()

low_quarter[ROLLING_VALUE] = low_quarter[STD_VALUE].rolling(12).mean()
low_quarter[ROLLING_MOMENTUM] = low_quarter[STD_MOMENTUM].rolling(12).mean()
low_quarter[ROLLING_QUALITY] = low_quarter[STD_QUALITY].rolling(12).mean()
low_quarter[ROLLING_VOLATILITY] = low_quarter[STD_VOLATILITY].rolling(12).mean()

plt.figure()
ax = plt.axes()
ax.plot(low_quarter[ROLLING_VALUE], color='red')
ax.plot(low_quarter[ROLLING_MOMENTUM], color='brown')
ax.plot(low_quarter[ROLLING_QUALITY], color='green')
ax.plot(low_quarter[ROLLING_VOLATILITY], color='blue')
plt.title('Low 25% RET_1, Rolling 12')
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()

# %% Model fit
p = 36  # Autoregressive
d = 1   # Integrated
q = 0   # Moving Average
order = (p, d, q)

# Rolling ARIMA of STD_VALUE
value_model = ARIMA(low_quarter[STD_VALUE], order=order)
value_model_fit = value_model.fit()
print(value_model_fit.summary())

# Rolling ARIMA of STD_MOMENTUM
momentum_model = ARIMA(low_quarter[STD_MOMENTUM], order=order)
momentum_model_fit = momentum_model.fit()
print(momentum_model_fit.summary())

# Rolling ARIMA of STD_QUALITY
quality_model = ARIMA(low_quarter[STD_QUALITY], order=order)
quality_model_fit = quality_model.fit()
print(quality_model_fit.summary())

# Rolling ARIMA of STD_VOLATILITY
volatility_model = ARIMA(low_quarter[STD_VOLATILITY], order=order)
volatility_model_fit = volatility_model.fit()
print(volatility_model_fit.summary())

# %% Plot residual
value_residuals = DataFrame(value_model_fit.resid, columns=[VALUE])
momentum_residuals = DataFrame(momentum_model_fit.resid, columns=[MOMENTUM])
quality_residuals = DataFrame(quality_model_fit.resid, columns=[QUALITY])
volatility_residuals = DataFrame(volatility_model_fit.resid, columns=[VOLATILITY])
residuals = pd.concat([value_residuals, momentum_residuals, quality_residuals, volatility_residuals])

plt.figure()
ax = plt.axes()
ax.plot(value_residuals, color='red', label=VALUE)
ax.plot(momentum_residuals, color='brown', label=MOMENTUM)
ax.plot(quality_residuals, color='green', label=QUALITY)
ax.plot(volatility_residuals, color='blue', label=VOLATILITY)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
residuals.plot(kind='kde')
plt.title('Residuals Histogram')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.show()

# %% Plot predict
value_predicts = DataFrame(value_model_fit.predict(typ='levels'), columns=[PRED_VALUE])
momentum_predicts = DataFrame(momentum_model_fit.predict(typ='levels'), columns=[PRED_MOMENTUM])
quality_predicts = DataFrame(quality_model_fit.predict(typ='levels'), columns=[PRED_QUALITY])
volatility_predicts = DataFrame(volatility_model_fit.predict(typ='levels'), columns=[PRED_VOLATILITY])
predicts = pd.concat([value_predicts, momentum_predicts, quality_predicts, volatility_predicts, low_quarter], axis=1)

plt.figure()
ax = plt.axes()
ax.plot(predicts[STD_VALUE], color='black')
ax.plot(predicts[PRED_VALUE], color='red')
plt.title(STD_VALUE + ' vs. ' + PRED_VALUE)
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
ax = plt.axes()
ax.plot(predicts[STD_MOMENTUM], color='black')
ax.plot(predicts[PRED_MOMENTUM], color='brown')
plt.title(STD_MOMENTUM + ' vs. ' + PRED_MOMENTUM)
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
ax = plt.axes()
ax.plot(predicts[STD_QUALITY], color='black')
ax.plot(predicts[PRED_QUALITY], color='green')
plt.title(STD_QUALITY + ' vs. ' + PRED_QUALITY)
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
ax = plt.axes()
ax.plot(predicts[STD_VOLATILITY], color='black')
ax.plot(predicts[PRED_VOLATILITY], color='blue')
plt.title(STD_VOLATILITY + ' vs. ' + PRED_VOLATILITY)
plt.xlabel('Date')
plt.ylabel('z-score')
plt.legend()
plt.grid(True)
plt.show()
