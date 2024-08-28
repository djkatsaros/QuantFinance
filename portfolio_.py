import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimzie as optimiz

days = 252
num_portfolios = 10000

class portfolio:
    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.start_ = start_date
        self.end = end_date

    def download_data(self):
        data = {}
        
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_, self.end_)
            data[stock] = ticker["Adj Close"]

        return pd.DataFrame(data)

    def get_adjclose(self):
        stock_data = self.download_data()
        stock_data = stock_data.resample('M').last()
        
        monthly = {}
        for i in range(0,len(self.stocks)):
            monthly[self.stocks[i]+'_adjclose' ] = stock_data[stocks[i]]

        self.monthly_data = pd.DataFrame(monthly)

    def regression(self,stock1,stock2):
        # use linear regression to fit a line to the data
        # slope is the 'beta' parameter in CAPM
        beta, alpha = polyfit(self.monthly_data[stock1],
                              self.monthly_data[stock2],
                              deg = 1)
        print("Beta from regression ", beta)
        # compute expected return according to CAPM formula, annualize!
        expected_return = Risk_free_rate + beta*(self.data['m_returns']*mos_in_year - Risk_free_rate)
        print("Expected return: ", expected_return)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize = (20,10))
        axis.scatter(self.data['m_returns'], self.data['s_returns'], label="Data points")
        axis.plot(self.data['m_returns'], beta*self.data['m_returns']+alpha,
                    color='red', label = 'CAPM line')
        plt.title('CAPM, finding alpha and beta')
        plt.xlabel('Market return $R_m$', fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08,0.05, '$R_a = \beta * R_m + \alpha$', fontsize =18)
        plt.legend()
        plt.grid(True)
        plt.show()
