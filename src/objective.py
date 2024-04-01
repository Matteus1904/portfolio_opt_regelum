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
        observation = argin[0][0]
        number_of_stocks = (len(observation) - 1)//2
        cash = observation[0]
        volume = observation[1:number_of_stocks+1]
        prices = observation[number_of_stocks+1:]
        total_revenue = rg.array([[volume.T@prices + cash]])
        action = argin[1]
        return super().__call__(-total_revenue, action, **kwargs)


class MarketRunningObjectiveModel(ModelQuadLin):
    def __init__(self, weights=rg.array([10])):
        super().__init__(
            quad_matrix_type="diagonal",
            weights=weights
        )

    def __call__(self, *argin, **kwargs):
        observation = argin[0][0]
        number_of_stocks = (len(observation) - 1)//2
        cash = observation[0]
        volume = observation[1:number_of_stocks+1]
        prices = observation[number_of_stocks+1:]
        total_revenue = rg.array([[volume.T@prices + cash]])
        action = argin[1]
        return super().__call__(total_revenue, action, **kwargs)