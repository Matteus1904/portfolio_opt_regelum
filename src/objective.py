from regelum.utils import rg
from torch import nn


class PortfolioRunningObjectiveModel(nn.Module):
    def __init__(self, weights=rg.array([10])):
        self.eps = 1.0e-5
        self.weights = weights
        super().__init__()

    def forward(self, *argin, **kwargs):
        state = argin[0][0]

        number_of_stocks = (len(state) - 3)//4
        current_cash = state[0]
        prev_cash = state[1]
        A = state[2]
        B = state[3]
        current_volumes = state[4:4+number_of_stocks]
        prev_volumes = state[4+number_of_stocks: 4+2*number_of_stocks]
        current_prices = state[4+2*number_of_stocks: 4+3*number_of_stocks]
        prev_prices = state[4+3*number_of_stocks: ]

        portfolio_return = ((current_prices.T)@ (current_volumes) - (prev_prices.T @ prev_volumes) + current_cash - prev_cash)/(prev_prices.T @ prev_volumes + prev_cash)
        D = rg.array([[(B*(portfolio_return - A) - (1/2)*A*(portfolio_return**2 - B))/(B-A**2 + self.eps)**(3/2)]])
        action = argin[1]
        return rg.array([(self.weights @ rg.concatenate((-D, action.T)))])
    

class MarketRunningObjectiveModel(nn.Module):
    def __init__(self, weights=rg.array([10])):
        self.eps = 1.0e-5
        self.weights = weights
        super().__init__()

    def forward(self, *argin, **kwargs):
        state = argin[0][0]

        number_of_stocks = (len(state) - 3)//4
        current_cash = state[0]
        prev_cash = state[1]
        A = state[2]
        B = state[3]
        current_volumes = state[4:4+number_of_stocks]
        prev_volumes = state[4+number_of_stocks: 4+2*number_of_stocks]
        current_prices = state[4+2*number_of_stocks: 4+3*number_of_stocks]
        prev_prices = state[4+3*number_of_stocks: ]

        portfolio_return = ((current_prices.T)@ (current_volumes) - (prev_prices.T @ prev_volumes) + current_cash - prev_cash)/(prev_prices.T @ prev_volumes + prev_cash)
        D = rg.array([[(B*(portfolio_return - A) - (1/2)*A*(portfolio_return**2 - B))/(B-A**2 + self.eps)**(3/2)]])
        action = argin[1]
        return rg.array([(self.weights @ rg.concatenate((D, action.T)))])
