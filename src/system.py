from regelum.system import System, ComposedSystem
import numpy as np
import casadi as cs
from regelum.utils import rg
from typing import Optional, Union, List
from typing_extensions import Self
from regelum.typing import RgArray
import torch

class ComposedSystem(ComposedSystem):
    def __init__(
        self,
        sys_left: Union[System, Self],
        sys_right: Union[System, Self],
        io_mapping: Optional[List[int]] = None,
        output_mode: str = "right",
        state_naming: List[str] = None,
        inputs_naming: List[str] = None,
        observation_naming: List[str] = None,
        action_bounds: List[List[float]] = None,
    ):
        self._step_size = None
        self._state_naming = (
            sys_left.state_naming + sys_right.state_naming if state_naming is None else state_naming
        )
        self._inputs_naming = (
            sys_left.inputs_naming + sys_right.inputs_naming if inputs_naming is None else inputs_naming
        )
        self._observation_naming = (
            sys_left.observation_naming + sys_right.observation_naming if observation_naming is None else observation_naming
        )
        self._action_bounds = (
            sys_left.action_bounds + sys_right.action_bounds if action_bounds is None else action_bounds
        )

        if io_mapping is None:
            io_mapping = np.arange(min(sys_left.dim_state, sys_right.dim_inputs))

        assert output_mode in [
            "state",
            "right",
            "both",
        ], "output_mode must be 'state', 'right' or 'both'"

        if "diff_eqn" in [sys_left.system_type, sys_right.system_type]:
            self._system_type = "diff_eqn"
        else:
            self._system_type = sys_right.system_type

        self.sys_left = sys_left
        self.sys_right = sys_right
        self._parameters = sys_left.parameters | sys_right.parameters
        self._dim_state = self.sys_right.dim_state + self.sys_left.dim_state
        if output_mode == "state":
            self._dim_observation = self.sys_left.dim_state + self.sys_right.dim_state
        elif output_mode == "right":
            self._dim_observation = self.sys_right.dim_observation
        elif output_mode == "both":
            self._dim_observation = (
                self.sys_left.dim_observation + self.sys_right.dim_observation
            )
        self.rout_idx, self.occupied_idx = self.__get_routing(io_mapping)
        self._dim_inputs = (
            self.sys_right.dim_inputs
            + self.sys_left.dim_inputs
            - len(self.occupied_idx)
        )
        self._name = self.sys_left.name + " + " + self.sys_right.name

        self.free_right_input_indices = np.setdiff1d(
            np.arange(self.sys_right.dim_inputs).astype(int),
            self.occupied_idx.astype(int),
        )
        self.output_mode = output_mode
        self.forward_permutation = np.arange(self.dim_observation).astype(int)
        self.inverse_permutation = np.arange(self.dim_observation).astype(int)

    def _compute_state_dynamics(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = True,
    ) -> RgArray:
        self.sys_left._step_size = self._step_size
        self.sys_right._step_size = self._step_size
        state = rg.array(state, prototype=state)
        inputs = rg.array(inputs, prototype=state)

        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        inputs_for_right = inputs[self.sys_left.dim_inputs:]
        dstate_of_right = rg.squeeze(
            self.sys_right.compute_state_dynamics(
                time=time,
                state=state_for_right,
                inputs=inputs_for_right,
                _native_dim=_native_dim,
            )
        )
        dstate_of_left = rg.squeeze(
            self.sys_left.compute_state_dynamics(
                time=time,
                state=rg.concatenate((state_for_left, state_for_right)),
                inputs=inputs_for_left,
                _native_dim=_native_dim
            )
        )
        final_dstate_vector = (
            rg.hstack((rg.force_row(dstate_of_left), rg.force_row(dstate_of_right)))
            if not isinstance(dstate_of_left, (np.ndarray, torch.Tensor))
            else rg.hstack((dstate_of_left, dstate_of_right))
        )

        assert (
            final_dstate_vector is not None
        ), f"final dstate_vector of system {self.name} is None"
        if not _native_dim:
            final_dstate_vector = rg.force_row(final_dstate_vector)
        return final_dstate_vector
    


    def _get_observation(
        self,
        time: Union[float, cs.MX],
        state: RgArray,
        inputs: RgArray,
        _native_dim: bool = False,
    ) -> RgArray:

        state = rg.array(state, prototype=state)
        inputs = rg.array(inputs, prototype=state)

        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        inputs_for_right = inputs[self.sys_left.dim_inputs:]
        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        outputs_of_left = self.sys_left.get_observation(
                time=time,
                state=rg.concatenate((state_for_left, state_for_right)),
                inputs=inputs_for_left,
                _native_dim=_native_dim,
        )

        outputs_of_right = self.sys_right.get_observation(
            time=time,
            state=state_for_right,
            inputs=inputs_for_right,
            _native_dim=_native_dim,
        )

        output = rg.concatenate((outputs_of_left.reshape(-1, 1), outputs_of_right.reshape(-1, 1)))


        return output

    
class Portfolio(System):
    r"""Implements the Portfolio optimizer"""
    _step_size: Optional[int] = None
    _name = "portfolio"
    _system_type = "discrete_stoch"

    def __init__(self, dim_state, action_bounds, transaction_cost):
        """Define number of stocks in the system"""
        self._dim_state = dim_state
        self.number_of_shares = (dim_state -3)//2
        self._dim_inputs = self.number_of_shares 
        self._dim_observation = self.number_of_shares +1
        if action_bounds is None:
            self._action_bounds = [[-1, 1]]*self._dim_inputs
        else:
            self._action_bounds = action_bounds
        self._state_naming = ['cash [USD]'] + ['first_momentum'] + ['second_momontum'] + [f'current_volume_{i}' for i in range(self._dim_inputs)]
        self._state_naming += [f'prev_volume_{i}' for i in range(self._dim_inputs)]
        self._observation_naming = ['cash share'] + [f'share_{i}' for i in range(self._dim_inputs)]
        self._inputs_naming =[f'delta_volume_{i}' for i in range(self._dim_inputs)]
        self.transaction_cost = transaction_cost
        super().__init__()
    
    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """

        Args:
            time: Current time.
            state: Current state.
            inputs: Current control inputs (i. e. action).

        Returns:
            Right-hand side of the portfolio.
        """

        number_of_stocks = (len(state) - 3)//4
        A = state[1]
        B = state[2]
        current_volumes = state[3:3+number_of_stocks]
        prev_volumes = state[3+number_of_stocks: 3+2*number_of_stocks]
        current_prices = state[3+2*number_of_stocks: 3+3*number_of_stocks]
        prev_prices = state[3+3*number_of_stocks: ]

        portfolio_return = ((current_prices.T)@ (current_volumes) - (prev_prices.T @ prev_volumes))/(prev_prices.T @ prev_volumes)

        Dstate = rg.zeros(self.dim_state, prototype=(state, inputs))
        prices_with_tr_cost = current_prices.copy()
        prices_with_tr_cost[inputs < 0] *= (1-self.transaction_cost)
        prices_with_tr_cost[inputs > 0] /= (1-self.transaction_cost)
        Dstate[0] = (- prices_with_tr_cost.T @ inputs)* self._step_size
        Dstate[1] = (1-self._step_size)*(portfolio_return - A)
        Dstate[2] = (1-self._step_size)*(portfolio_return**2  - B)
        Dstate[3:self.number_of_shares+3] = inputs.reshape(1, -1) * self._step_size
        Dstate[self.number_of_shares+3:] = (current_volumes - prev_volumes).reshape(1, -1)
        return Dstate
    

    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
        current_prices = state[2*self.number_of_shares+3:3*self.number_of_shares+3]
        current_volumes = state[3:self.number_of_shares+3]

        cash = state[0]
        cash_and_volumes = rg.concatenate((rg.array([cash]), current_volumes))
        observation =(rg.concatenate((rg.array([[1.]]), current_prices)) * cash_and_volumes) / (rg.concatenate((rg.array([[1.]]), current_prices)).T @ cash_and_volumes)
        return observation

    
class MarketAttack(System):
    r"""Implements the market attack against portfolio optimizer"""
    _step_size: Optional[int] = None
    _name = "market"
    _system_type = "discrete_stoch"

    def __init__(self, dim_state, action_bounds):
        """Initialize Market depending of number of stocks in portfolio"""
        self._dim_state = dim_state
        self.number_of_shares = dim_state//2
        self._dim_inputs = 2*self.number_of_shares + self.number_of_shares*(self.number_of_shares-1)//2
        self._dim_observation = self.number_of_shares
        if action_bounds is None:
            self._action_bounds = [[-0.05, 0.05]]*self._dim_inputs
        else:
            self._action_bounds = action_bounds
        self._observation_naming = [f'returns_{i}' for i in range(self.number_of_shares)]
        self._state_naming = [f'current_price_{i} [USD]' for i in range(self.number_of_shares)] + [f'prev_price_{i} [USD]' for i in range(self.number_of_shares)]
        self._inputs_naming =[f'drift_{i}' for i in range(self.number_of_shares)] + [f'volatility_{i}' for i in range(self.number_of_shares)]
        self._inputs_naming += [f'corr_{i}_{j}' for i in range(self.number_of_shares) for j in range(i+1, self.number_of_shares)]
        super().__init__()

    def _compute_state_dynamics(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ) -> RgArray:
        """
        Args:
            time: Current time.
            state: Current state.
            inputs: Current control inputs (i. e. action).

        Returns:
            Right-hand side of market.
        """
        current_prices, prev_prices = state[:self.number_of_shares], state[self.number_of_shares:]
        drifts = inputs[:self.number_of_shares]
        variances = inputs[self.number_of_shares:2*self.number_of_shares]
        corr = inputs[2*self.number_of_shares:].flatten()
        corr_matrix = rg.zeros((self.number_of_shares, self.number_of_shares))
        corr_matrix[np.triu_indices(self.number_of_shares, k=1)] = corr
        corr_matrix = corr_matrix + corr_matrix.T + np.eye(self.number_of_shares)
        eigval, eigvec = np.linalg.eig(corr_matrix)
        if not all(eigval>0):
            val = np.maximum(eigval,1e-5)
            B = np.diag(np.sqrt(1/(eigvec * eigvec @ val)))@eigvec@np.diag(np.sqrt(val))
            corr_matrix = B@B.T
        
        choleskyMatrix = np.linalg.cholesky(corr_matrix)
        cX = rg.dot(choleskyMatrix, np.random.normal(size = (self.number_of_shares))).reshape(-1, 1)

        Dstate = rg.concatenate((drifts * (self._step_size)*current_prices + variances*rg.sqrt(self._step_size)*current_prices*cX, current_prices - prev_prices))

        return Dstate
    
    def _get_observation(
        self, time: Union[float, cs.MX], state: RgArray, inputs: RgArray
    ):
        current_prices, prev_prices = state[:self.number_of_shares], state[self.number_of_shares:]
        returns = (current_prices - prev_prices)/prev_prices
        return returns

    

