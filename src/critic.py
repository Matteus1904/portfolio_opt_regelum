from regelum.critic import Critic
import numpy as np
from regelum.optimizable import Optimizable


from regelum.model import Model, ModelNN
from typing import Optional, Union, List
from regelum.optimizable import OptimizerConfig
from regelum.__internal.base import apply_callbacks

class Critic(Critic):

    def __init__(
        self,
        model: Union[Model, ModelNN],
        system_dim_observation: int = 1,
        system_dim_inputs: int = 1,
        td_n: int = 1,
        device: Union[str] = "cpu",
        is_same_critic: bool = False,
        is_value_function: bool = False,
        is_on_policy: bool = False,
        optimizer_config: Optional[OptimizerConfig] = None,
        regularization_param: float = 0.0,
        action_bounds: Optional[Union[List, np.array]] = None,
        size_mesh: Optional[int] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        is_full_iteration_epoch: bool = False,
    ):

        Optimizable.__init__(self, optimizer_config=optimizer_config)

        self.model = model
        self.discount_factor = discount_factor
        self.system_dim_observation = system_dim_observation
        self.system_dim_inputs = system_dim_inputs
        self.sampling_time = sampling_time
        self.td_n = td_n
        self.device = device
        self.is_same_critic = is_same_critic
        self.is_value_function = is_value_function
        self.is_on_policy = is_on_policy
        self.action_bounds = (
            np.array(action_bounds) if action_bounds is not None else None
        )
        self.regularization_param = regularization_param
        self.size_mesh = size_mesh
        self.is_full_iteration_epoch = is_full_iteration_epoch

        self.initialize_optimize_procedure()

    def initialize_optimize_procedure(self):
        """Instantilize optimization procedure via Optimizable functionality."""
        self.batch_size = self.get_data_buffer_batch_size()
        (
            self.running_objective_var,
            self.critic_targets_var,
            self.critic_weights_var,
            self.critic_stored_weights_var,
        ) = (
            self.create_variable(
                self.batch_size,
                1,
                name="running_objective",
                is_constant=True,
            ),
            self.create_variable(
                self.batch_size,
                1,
                name="critic_targets",
                is_constant=True,
            ),
            self.create_variable(
                name="critic_weights",
                like=self.model.named_parameters,
            ),
            self.create_variable(
                name="critic_stored_weights",
                like=self.model.cache.weights,
                is_constant=True,
            ),
        )
        if not self.is_value_function:
            self.observation_action_var = self.create_variable(
                self.batch_size,
                self.system_dim_observation + self.system_dim_inputs,
                name="observation_action",
                is_constant=True,
            )
            self.critic_model_output = self.create_variable(
                self.batch_size,
                1,
                name="critic_model_output",
                is_nested_function=True,
                nested_variables=[
                    self.observation_action_var,
                    self.critic_stored_weights_var,
                    self.critic_weights_var,
                ],
            )
        else:
            self.observation_var = self.create_variable(
                self.batch_size,
                self.system_dim_observation,
                name="observation",
                is_constant=True,
            )
            self.critic_model_output = self.create_variable(
                self.batch_size,
                1,
                name="critic_model_output",
                is_nested_function=True,
                nested_variables=[
                    self.observation_var,
                    self.critic_stored_weights_var,
                    self.critic_weights_var,
                ],
            )

        if hasattr(self.model, "weight_bounds"):
            self.register_bounds(self.critic_weights_var, self.model.weight_bounds)

        if self.is_value_function:
            self.connect_source(
                connect_to=self.critic_model_output,
                func=self.model,
                source=self.observation_var,
                weights=self.critic_weights_var,
            )
            if (not self.is_same_critic) and self.is_on_policy:
                self.connect_source(
                    connect_to=self.critic_targets_var,
                    func=self.model.cache,
                    source=self.observation_var,
                    weights=self.critic_stored_weights_var,
                )
        else:
            self.connect_source(
                connect_to=self.critic_model_output,
                func=self.model,
                source=self.observation_action_var,
                weights=self.critic_weights_var,
            )
            if (not self.is_same_critic) and self.is_on_policy:
                self.connect_source(
                    connect_to=self.critic_targets_var,
                    func=self.model.cache,
                    source=self.observation_action_var,
                    weights=self.critic_stored_weights_var,
                )

        self.episode_id_var = self.create_variable(
            self.batch_size,
            1,
            name="episode_id",
            is_constant=True,
        )

        self.register_objective(
            func=self.objective_function,
            variables=[
                self.episode_id_var,
                self.running_objective_var,
                self.critic_model_output,
                self.critic_weights_var,
                self.critic_stored_weights_var,
            ]
            + (
                [self.critic_targets_var]
                if ((not self.is_same_critic) or (not self.is_on_policy))
                else []
            ),
        )
    @apply_callbacks()
    def optimize(
        self, data_buffer, is_update_and_cache_weights=True, is_constrained=True
    ):
        if isinstance(self.model, ModelNN):
            self.model.to(self.device)
            self.model.cache.to(self.device)

        if not self.is_on_policy:
            self.update_data_buffer_with_optimal_policy_targets(data_buffer)

        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )
        weights = None
        if opt_kwargs is not None:
            if self.kind == "tensor":
                super().optimize(**opt_kwargs, is_constrained=is_constrained)
                if is_update_and_cache_weights:
                    self.model.update_and_cache_weights()
            elif self.kind == "symbolic":
                weights = super().optimize(
                    **opt_kwargs,
                    is_constrained=is_constrained,
                    critic_weights=self.model.weights,
                    critic_stored_weights=self.model.cache.weights,
                )["critic_weights"]
                if is_update_and_cache_weights:
                    self.model.update_and_cache_weights(weights)

        if not self.is_on_policy:
            self.delete_critic_targets(data_buffer)

        return weights