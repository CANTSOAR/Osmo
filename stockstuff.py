import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

def create_data(stock, btrange = 12, interval = "1d"):
    #range is in months

    ticker = yf.Ticker(stock)
    start = dt.date.today() - dt.timedelta(days = 30 * btrange)
    stock_data = ticker.history(interval = interval, start = start, prepost=False)

    close_prices = []
    open_prices = []

    for x in range(len(stock_data["Volume"])):
        if stock_data["Volume"][x] != 0:
            close_prices.append(stock_data["Close"][x])
            open_prices.append(stock_data["Open"][x])

    return np.array(stock_data["Close"]), np.array(close_prices), np.array(open_prices), np.array(stock_data["Volume"])

def calc_rsis(prices, period = 14):
    prices = np.array(prices)

    price_changes = prices[1:] - prices[:-1]

    initial_avg_gain = np.sum(price_changes[:period] * (price_changes[:period] > 0)) / period
    initial_avg_loss = -np.sum(price_changes[:period] * (price_changes[:period] < 0)) / period

    avg_gains = [initial_avg_gain]
    avg_losses = [initial_avg_loss]

    initial_rsi = 100 - 100 / (1 + initial_avg_gain / initial_avg_loss)

    rsis = [initial_rsi]

    for index in range(len(prices) - period):
        avg_gains.append((avg_gains[index] * (period - 1) + price_changes[index + period - 1] * (price_changes[index + period - 1] > 0)) / period)
        avg_losses.append((avg_losses[index] * (period - 1) - price_changes[index + period - 1] * (price_changes[index + period - 1] < 0)) / period)
        rsis.append(100 - 100 / (1 + avg_gains[index + 1] / avg_losses[index + 1]))

    return rsis

def calc_macd(prices, fast = 12, slow = 26, signal = 9):
    macd = calc_ema(prices, fast) - calc_ema(prices, slow)
    macd_signal = calc_ema(macd, signal)

    normalized_macd = [.5 for i in range(20)]
    for i in range(len(macd) - 20):
        normalized_macd.append(macd[i + 20] / max(abs(macd[i:i+20])))

    return macd, macd_signal, normalized_macd

def calc_ema(prices, period = 14):

    initial_ema = np.mean(prices[:period])

    emas = []
    emas.append(initial_ema)

    smoothing_factor = 2 / (1 + period)

    for index in range(1, len(prices) - period):
        ema = prices[index + period] * smoothing_factor + emas[index - 1] * (1 - smoothing_factor)
        emas.append(ema)

    emas = list(prices[:period]) + emas

    return np.array(emas)

def sharpe(stocks, ratios, rf = .04, btrange = 12, interval = "1d", show = False):
    ratios = np.array(ratios)

    if sum(abs(ratios)) != 1:
        print("Portfolio Ratios Mismatch")
        return
    
    closedata = []

    for stock in stocks:
        temp = create_data(stock, btrange, interval)
        closedata.append(temp[2])

    closedata = np.array(closedata)

    possible_intervals = ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1mo","3mo","6mo","1y","2y","5y","10y"]
    interval_coefficients = [252 * 13 * 15 * 1, 252 * 13 * 15, 252 * 26 * 3, 252 * 26, 252 * 13, 252 * 7, 252 * 5, 252 * 7, 252, 52, 12, 4, 2, 1, .5, .2, .1]

    changes = np.array([(closedata[stock][1:] - closedata[stock][:-1]) / closedata[stock][:-1] for stock in range(len(closedata))])

    sds = [np.std(changes[stock]) * interval_coefficients[possible_intervals.index(interval)]**(1/2) for stock in range(len(changes))]
    ratioed_sd = np.dot(ratios, np.array(sds))
    ratioed_sd = (np.dot(ratioed_sd, ratioed_sd.T) * interval_coefficients[possible_intervals.index(interval)])**(1/2)
    returns = [(abs(closedata[stock][-1] - closedata[stock][0]) / closedata[stock][0])**(12/btrange) * (1 - 2 * bool(closedata[stock][0] > closedata[stock][-1])) for stock in range(len(closedata))]

    sharpes = [(returns[stock] - rf) / sds[stock] for stock in range(len(returns))]
    ratioed_sharpe = (np.dot(np.array(returns), ratios.T) - rf) / ratioed_sd

    if show:
        print("Individual Sharpes:", sharpes)
        print("Ratio Accounted Sharpe:", ratioed_sharpe)

    return ratioed_sharpe

def plot_simple_holdings(stocks, ratios, btrange = 12, interval = "1d"):
    ratios = np.array(ratios)

    if sum(abs(ratios)) != 1:
        print("Portfolio Ratios Mismatch")
        return
    
    closedata = []

    for stock in stocks:
        temp = create_data(stock, btrange, interval)
        closedata.append(temp[2])

    closedata = np.array(closedata)
    changes = np.array([[(closedata[stock][i] - closedata[stock][0]) / closedata[stock][0] for i in range(len(closedata[0]))] for stock in range(len(closedata))])
    ratioed_data = np.dot(ratios, changes)

    returns = [(abs(closedata[stock][-1] - closedata[stock][0]) / closedata[stock][0])**(12/btrange) * (1 - 2 * bool(closedata[stock][0] > closedata[stock][-1])) for stock in range(len(changes))]
    ratioed_return = np.dot(np.array(returns), ratios)

    colors = color_gradient(len(changes))

    print("Individal Returns: ", returns)
    print("Ratio Accounted Return:", ratioed_return)

    for stock, color in zip(changes, colors):
        plt.plot(range(len(stock)), stock, color = color)
    plt.plot(range(len(ratioed_data)), ratioed_data, color = "black")
    plt.plot([0, len(ratioed_data)], [0, ratioed_data[-1]], color = "maroon")
    plt.show()

def color_gradient(points):
  colors = []

  for point in range(points):
    colors.append((point/points, 1, 1))

  return clrs.hsv_to_rgb(colors)
