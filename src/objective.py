from regelum.model import (
    ModelQuadLin
)
from regelum.utils import rg

class PortfolioRunningObjectiveModel(ModelQuadLin):
    def __init__(self, weights=rg.array([10])):
        super().__init__(
            quad_matrix_type="diagonal",
            weights=weights
        )
        self.eps = 1.0e-5

    def __call__(self, *argin, **kwargs):
        assert argin[0][0].shape == (19, )
        state = argin[0][0]

        number_of_stocks = (len(state) - 3)//4
        cash = state[0]
        A = state[1]
        B = state[2]
        current_prices = state[3:3+number_of_stocks]
        prev_prices = state[3+number_of_stocks: 3+2*number_of_stocks]
        current_volumes = state[3+2*number_of_stocks: 3+3*number_of_stocks]
        prev_volumes = state[3+3*number_of_stocks: ]
        portfolio_return = ((current_prices.T)@ (current_volumes) - (prev_prices.T @ prev_volumes))/(prev_prices.T @ prev_volumes)
        D = rg.array([[(B*(portfolio_return - A) - (1/2)*A*(portfolio_return**2 - B))/(B-A**2 + 1e-5)**(3/2)]])
        action = argin[1]
        return super().__call__(1/D, action, **kwargs)


class MarketRunningObjectiveModel(ModelQuadLin):
    def __init__(self, weights=rg.array([10])):
        super().__init__(
            quad_matrix_type="diagonal",
            weights=weights
        )

    def __call__(self, *argin, **kwargs):
        assert argin[0][0].shape == (19, )
        state = argin[0][0]

        number_of_stocks = (len(state) - 3)//4
        cash = state[0]
        A = state[1]
        B = state[2]
        current_prices = state[3:3+number_of_stocks]
        prev_prices = state[3+number_of_stocks: 3+2*number_of_stocks]
        current_volumes = state[3+2*number_of_stocks: 3+3*number_of_stocks]
        prev_volumes = state[3+3*number_of_stocks: ]
        portfolio_return = ((current_prices.T)@ (current_volumes) - (prev_prices.T @ prev_volumes))/(prev_prices.T @ prev_volumes)
        D = rg.array([[(B*(portfolio_return - A) - (1/2)*A*(portfolio_return**2 - B))/(B-A**2 + 1e-5)**(3/2)]])
        action = argin[1]
        return super().__call__(D, action, **kwargs)