"""
Implements Markowitz portfolio optimization model.

Chooses the portfolio with the best return for a given risk by optimizing the sharpe ratio 
[https://en.wikipedia.org/wiki/Sharpe_ratio] :
E[R_a - R_f]/sigma_a
for R_a the asset return (computed), R_f the risk free return (computed with given risk free interest rate) 
and sigma_a the risk (standard deviation) of the asset.
Essentially gives a net return / unit risk measurement.


"""
import pandas as pd
import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt 
import scipy.optimize as optimization 

#
# Global parameters
#

# start and end dates
start_date = '2012-01-01'
end_date = '2017-01-01'
# list of stocks
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
days = 252
NUM_portfolios = 10000

class portfolio:
    def __init__(self, stocks, days, NUM_portfolios, start_date, end_date):
        self.stocks = stocks
        self.start = start_date
        self.end = end_date

    def download_data(self):
        # name of the stock (key) : stock values (values)
        stock_data = {}
    
        for stock in self.stocks:
            # closing prices
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(start = start_date, end = end_date)['Close']
            
        return pd.DataFrame(stock_data)

    def show_data(self):
        self.data.plot(figsize = (9,4))
        plt.show()

    def calc_return(self):
        # use log to compare the returns on a comparable scale
        log_return = np.log(self.data/self.data.shift(1))
        return log_return[1:]

    def show_stats(self):
        """prints mean and covariance matrix for a dataframe"""
        print(self.data.mean() * days) # mean annual return
        print(self.data.cov() * days) # cov of annual returns (matrix)

    def show_mean_var(self):
        # compute mean/variance of a portfolio
        returns = self.calc_return()
        portfolio_return = np.sum(returns.mean() * self.weights) * days  #portfolio_return = np.sum(self.data.mean()*self.weights)*days # annualized
        # volatility = w^T \cdot \Sigma \cdot w (\Sigma the covar. matrix of the portfolio) 
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(returns.cov()
                                                            * NUM_TRADING_DAYS, self.weights)))
        #portfolio_volatility = np.sqrt(np.dot(self.weights.T, 
        #                                      np.dot(self.data.cov()*days,
        #                                             self.weights)))
        print("Expected portfolio mean (return):", portfolio_return)
        print("Expected portfolio volatility (st dev):", portfolio_volatility)

    def generate_portfolios(self):
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []
        returns= self.calc_return()
        for _ in range(NUM_portfolios):
            w = np.random.random(len(stocks))
            w /= np.sum(w)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(returns.mean() * w) * 252)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                          * 252, w))))
        return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

    def show_portfolios(self):
        plt.figure(figsize= (10,6))
        plt.scatter(self.risks, self.data, c=self.data/self.risks, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

    def stats(self):
        portfolio_return = np.sum(self.data.mean() * self.weights) * days
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.data.cov()
                                                                * days, self.weights)))
        return np.array([portfolio_return, portfolio_volatility,
                         portfolio_return / portfolio_volatility])

    def optimize_portfolio(self):
        daily_returns = self.calc_return()
        def min_func__sharpe(self,x):
            #daily_returns = self.log_daily_returns
            return - (np.sum(daily_returns.mean() * x) * days) / (np.sqrt(np.dot(x, np.dot(daily_returns.cov() * days, x))))
        # constraints: sum of weights = 1
        # f(x) = 0, this is the fucntion to minimize (hence we subtract 1)
        constraints = ({ 'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        # weights can be at most 1 (1 corr.s to 100% investment in that stock)
        bounds = tuple((0,1) for _ in range(len(stocks)))
        return optimization.minimize(fun = min_func__sharpe, x0=self.weights[0],
                                     args = daily_returns,
                                     method = 'SLSQP',
                                     bounds = bounds,
                                     constraints = constraints)

    def print_opt_portfolio(self):
        print("Optimal portfolio: ", optimum['x'].round(3))
        print("Expected return, volatility and Sharpe ratio: ",
              self.stats(self.optimum['x'].round(3), self.data))

    def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(self.stats(opt['x'], rets)[1], self.stats(opt['x'], rets)[0], 'g*', markersize=20.0)
        plt.show()

    
    def initialize(self):
        self.data = self.download_data()
        #elf.calc_return()
        self.weights, self.means, self.risks = self.generate_portfolios()
        #self.log_daily_returns = self.calc_return() 
        self.optimum = self.optimize_portfolio()

#    if __name__ == '__main__':
 #       dataset = download_data()
   #     show_data(dataset)
  #      log_daily_returns = calc_return(dataset)

    #    pweights, means, risks = generate_portfolios(log_daily_returns)
      #  show_portfolios(means, risks)
     #   optimum = optimize_portfolio(pweights, log_daily_returns)
       # print_opt_portfolio(optimum, log_daily_returns)
        #show_optimal_portfolio(optimum, log_daily_returns, means, risks)

