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
        state = state[self.forward_permutation]

        state_for_left, state_for_right = (
            state[: self.sys_left.dim_state],
            state[self.sys_left.dim_state :],
        )
        inputs_for_left = inputs[: self.sys_left.dim_inputs]
        dstate_of_left = rg.squeeze(
            self.sys_left.compute_state_dynamics(
                time=time,
                state=(state_for_left, state_for_right),
                inputs=inputs_for_left,
                _native_dim=_native_dim
            )
        )
        outputs_of_left = rg.squeeze(
            self.sys_left.get_observation(
                time=time,
                state=state_for_left,
                inputs=inputs_for_left,
                _native_dim=_native_dim,
            )
        )

        inputs_for_right = rg.zeros(
            self.sys_right.dim_inputs,
            prototype=(state, inputs),
        )
        if len(self.occupied_idx) > 0:
            inputs_for_right[self.occupied_idx] = outputs_of_left[self.rout_idx]
        inputs_for_right[self.free_right_input_indices] = rg.reshape(
            inputs[self.sys_left.dim_inputs :],
            rg.shape(inputs_for_right[self.free_right_input_indices]),
        )

        dstate_of_right = rg.squeeze(
            self.sys_right.compute_state_dynamics(
                time=time,
                state=state_for_right,
                inputs=inputs_for_right,
                _native_dim=_native_dim,
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
        final_dstate_vector = final_dstate_vector[self.inverse_permutation]
        if not _native_dim:
            final_dstate_vector = rg.force_row(final_dstate_vector)
        return final_dstate_vector
    
    
class Portfolio(System):
    r"""Implements the Portfolio optimizer"""
    _step_size: Optional[int] = None
    _name = "portfolio"
    _system_type = "discrete_stoch"

    def __init__(self, dim_state, action_bounds):
        """Define number of stocks in the system"""
        self._dim_state = dim_state
        self._dim_inputs = dim_state - 1 
        self._dim_observation = dim_state
        if action_bounds is None:
            self._action_bounds = [[-1, 1]]*self._dim_inputs
        else:
            self._action_bounds = action_bounds
        self._observation_naming = self._state_naming = ['cash [USD]'] + [f'volume_{i}' for i in range(dim_state - 1)]
        self._inputs_naming =[f'delta_volume_{i}' for i in range(self._dim_inputs)]
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
        prices = state[1]
        volumes = state[0] 
        Dstate = rg.zeros(self.dim_state, prototype=(state, inputs))
        Dstate[0] = prices.T @ inputs
        Dstate[1:] = inputs.reshape(1, -1)
        return Dstate
    
class MarketAttack(System):
    r"""Implements the market attack against portfolio optimizer"""
    _step_size: Optional[int] = None
    _name = "market"
    _system_type = "discrete_stoch"

    def __init__(self, dim_state, action_bounds):
        """Initialize Market depending of number of stocks in portfolio"""
        self._dim_state = dim_state
        self._dim_inputs = 2*dim_state + dim_state*(dim_state-1)//2
        self._dim_observation = dim_state
        if action_bounds is None:
            self._action_bounds = [[0.05, 0.1]]*self._dim_inputs
        else:
            self._action_bounds = action_bounds
        self._observation_naming = self._state_naming = [f'price_{i} [USD]' for i in range(dim_state)]
        self._inputs_naming =[f'drift_{i}' for i in range(dim_state)] + [f'volatility_{i}' for i in range(dim_state)]
        self._inputs_naming += [f'corr_{i}_{j}' for i in range(dim_state) for j in range(i+1, dim_state)]
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
        prices = state.reshape(1, -1)
        drifts = inputs[:self.dim_state]
        variances = inputs[self.dim_state:2*self.dim_state]
        corr = inputs[2*self.dim_state:].flatten()
        corr_matrix = rg.zeros((self.dim_state, self.dim_state))
        corr_matrix[np.triu_indices(self.dim_state, k=1)] = corr
        corr_matrix = corr_matrix + corr_matrix.T + np.eye(self.dim_state)
        choleskyMatrix = np.linalg.cholesky(corr_matrix)
        cX = rg.dot(choleskyMatrix, np.random.normal(size = (self.dim_state)))

        Dstate = rg.array(drifts * (self._step_size)*prices + variances*rg.sqrt(self._step_size)*drifts*prices*cX)

        return Dstate
    

