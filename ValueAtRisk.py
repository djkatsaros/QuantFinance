"""
Computes value at risk measure for a given (collection of) assets.
Takes user inputed number of days into the future (1 = tomorrow),
confidence interval, and number of shares ('position').

Prints out the VaR for the given inputs. VaR is a number is in a currency units
which should be understood as meaning
"With c*100 % confidence we will not lose more money than $ VaR in n days". 
"""

import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    """
    Downloads data
      > stock: Tickers
      > start_date: date in 'yyyy-mm-dd'
      > end_date:   similar to start^^^
    """
    data = {}
    for st in stock:
        ticker = yf.download(st, start_date, end_date)
        data[st] = ticker['Adj Close']
    return pd.DataFrame(data)


 
def calculate_var(pos, c, mu, sig):
    """ 
    Calculates value at risk using the variance method for 1 day
      > pos: i.e. the investment  
      > c:   confidence interval
      > mu:  mean for the stock return distribution
      > sig: standard dev. for the stock return distribution
      Use the equation:
            VaR = \Delta S * (\mu*[\delta t] - \sigma * [\delta t]^{1/2} * alpha (1-c)
        where alpha is the inverse cdf for the standard normal distribution, implemented 
        with the .ppf method from scipy stats
    """
    var = pos * (mu - sig * norm.ppf(1-c))
    return var


def calculate_var_n(position, c, mu, sigma, n):
    """
    Similar to calculate_var. Applies the function for multiple day.
    ONly new variable is n, the number of days.
    """
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1-c))
    return var


if __name__ == '__main__':
    # Main 
    """
    Collects user inputs. Executes computations and prints outputs. 
    Includes some checks on user inputs.
    """
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2021, 1, 1)
    stocks = ['C','DB','AAPL'] #TODO, make user input. 
    stock_data = download_data(stocks, start, end)
    for st in stocks:
        stock_data[st+' returns'] = np.log(stock_data[st] / stock_data[st].shift(1))
        stock_data = stock_data[1:]
    print(stock_data)

    # this is the investment (stocks or whatever)
    S = float(input('Enter the position (number of shares etc.): ')) 
    # confidence level. Typically between 90 and 99%
    c = float(input('Enter the confidence level: '))
    if c > 1:
        c /= 100
    elif c < 0:
        c = float(input('Confidence interval needs to be betwen 0 and 1. Enter confidence level: '))
    
    n = float(input("Enter number of days into the future (1 means you want VaR tomorrow): "))
    if n != np.floor(n) and n != np.ceil(n):
        n = float(input("Number of days needs to be a whole number (we are considering closing prices only in this model). Enter a whole number of days: "))

    for st in stocks:
    # we assume that daily returns are normally distributed
        mu = np.mean(stock_data[st+' returns'])
        sigma = np.std(stock_data[st+' returns'])
        
        # "With c*100 % confidence we will not lose more money than $ VaR in n days". 
        print('Value at risk %f days into the future for %s is: $%0.2f' % (n, st, calculate_var_n(S, c, mu, sigma, n)))




