from regelum.policy import Policy
from regelum.critic import Critic
from regelum.model import (
    PerceptronWithTruncatedNormalNoise,
    ModelNN
)
from regelum.system import ComposedSystem
import torch as th
from regelum.data_buffers.batch_sampler import RollingBatchSampler
from regelum.optimizable.core.configs import TorchOptimizerConfig
import numpy as np
from regelum.data_buffers import DataBuffer
from regelum.utils import rg
from regelum.__internal.base import apply_callbacks

def get_gae_advantage(
    gae_lambda: float,
    running_objectives: th.FloatTensor,
    values: th.FloatTensor,
    times: th.FloatTensor,
    discount_factor: float,
    sampling_time: float,
) -> th.FloatTensor:
    deltas = (
        running_objectives[:-1]
        + discount_factor**sampling_time * values[1:]
        - values[:-1]
    )
    if gae_lambda == 0.0:
        advantages = deltas
    else:
        gae_discount_factors = (gae_lambda * discount_factor) ** times[:-1]
        reversed_gae_discounted_deltas = th.flip(
            gae_discount_factors * deltas, dims=[0, 1]
        )
        advantages = (
            th.flip(reversed_gae_discounted_deltas.cumsum(dim=0), dims=[0, 1])
            / gae_discount_factors
        )
    return advantages


def vpg_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: th.FloatTensor,
    actions: th.FloatTensor,
    times: th.FloatTensor,
    episode_ids: th.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: th.FloatTensor,
    sampling_time: float,
    is_normalize_advantages: bool,
    gae_lambda: float,
) -> th.FloatTensor:
    critic_values = critic_model(observations)
    log_pdfs = policy_model.log_pdf(observations, actions).reshape(-1)

    objective = 0.0
    for episode_idx in th.unique(episode_ids):
        mask = (episode_ids == episode_idx).reshape(-1)
        advantages = get_gae_advantage(
            gae_lambda=gae_lambda,
            running_objectives=running_objectives[mask],
            values=critic_values[mask],
            times=times[mask],
            discount_factor=discount_factor,
            sampling_time=sampling_time,
        ).reshape(-1)

        if is_normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        objective += (
            discount_factor ** times[mask][:-1].reshape(-1)
            * advantages
            * log_pdfs[mask][:-1]
        ).sum()

    return objective / N_episodes


class JointPolicyVPG(Policy):
    def __init__(
        self,
        pedestrian_model: PerceptronWithTruncatedNormalNoise,
        chauffeur_model: PerceptronWithTruncatedNormalNoise,
        pedestrian_critic: Critic,
        chauffeur_critic: Critic,
        system: ComposedSystem,
        is_normalize_advantages: bool = True,
        gae_lambda: float = 0.95,
        N_episodes: int = 1,
        sampling_time: float = 0.1,
    ):
        def freeze_stds(params):
            for p in params():
                if p[0] == "stds":
                    p[1].requires_grad_(False)
            return params  # Detaches parameters named "stds"

        iter_batches_kwargs = {
            "batch_sampler": RollingBatchSampler,
            "dtype": th.FloatTensor,
            "mode": "full",
            "n_batches": 1,
            "device": "cpu",
        }
        super().__init__(
            system=system,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=1,  # Only one grad step
                data_buffer_iter_bathes_kwargs=iter_batches_kwargs,
                opt_method_kwargs=dict(lr=1e-3),
            ),
            action_bounds=system.sys_left.action_bounds
            + system.sys_right.action_bounds,
        )
        self.is_normalize_advantages = is_normalize_advantages
        self.gae_lambda = gae_lambda
        self.pedestrian_model = pedestrian_model
        self.chauffeur_model = chauffeur_model
        self.model_to_optimize = self.pedestrian_model

        self.pedestrian_critic = pedestrian_critic
        self.chauffeur_critic = chauffeur_critic
        self.current_critic = self.pedestrian_critic
        self.N_episodes = N_episodes
        self.sampling_time = sampling_time

        ## Define an optimization problem here

        self.pedestrian_model_weigths = self.create_variable(
            name="pedestrian_model_weights", like=self.pedestrian_model.named_parameters
        )
        self.pedestrian_model_weigths.register_hook(freeze_stds)

        self.chauffeur_model_weights = self.create_variable(
            name="chauffeur_model_weights",
            like=self.chauffeur_model.named_parameters,
            is_constant=True,
        )
        self.chauffeur_model_weights.register_hook(freeze_stds)

        self.objective_inputs = [
            self.create_variable(name=variable_name, is_constant=True)
            for variable_name in self.data_buffer_objective_keys()
        ]
        self.register_objective(
            self.objective_function, variables=self.objective_inputs
        )

    def switch_critic(self):
        # This method will be triggered from Scenario
        self.current_critic = (
            self.pedestrian_critic
            if self.current_critic is self.chauffeur_critic
            else self.chauffeur_critic
        )

    def switch_model_to_optimize(self):
        # This method will be triggered from Scenario
        self.model_to_optimize = (
            self.pedestrian_model
            if self.model_to_optimize is self.chauffeur_model
            else self.chauffeur_model
        )
        return self.model_to_optimize

    def action_col_idx(self):
        return (
            slice(0, self.system.sys_left.dim_inputs)
            if self.model_to_optimize is self.pedestrian_model
            else slice(self.system.sys_right.dim_inputs, None)
        )

    def objective_function(
        self,
        observation: th.Tensor,  # All arguments here will be passed from data buffer
        action: th.Tensor,
        time: th.Tensor,
        episode_id: th.Tensor,
        running_objective: th.Tensor,
    ):
        actions_of_current_model = action[
            :, self.action_col_idx()
        ]  # Choose respective actions
        return vpg_objective(
            policy_model=self.model_to_optimize,
            critic_model=self.current_critic.model,
            observations=observation,
            actions=actions_of_current_model,
            times=time,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
            sampling_time=self.sampling_time,
            is_normalize_advantages=self.is_normalize_advantages,
            gae_lambda=self.gae_lambda,
        )

    def get_action(self, observation: np.array) -> np.array:
        action_pedestrian = self.pedestrian_model(th.FloatTensor(observation))
        action_chauffeur = self.chauffeur_model(th.FloatTensor(observation))
        action = rg.hstack((action_pedestrian, action_chauffeur)).detach().cpu().numpy()
        return action  # Concatenate actions in order to pass them as a whole into Scenario at runtime

    def data_buffer_objective_keys(self):
        return ["observation", "action", "time", "episode_id", "running_objective"]

    @apply_callbacks()
    def optimize(self, data_buffer: DataBuffer) -> None:
        opt_kwargs = data_buffer.get_optimization_kwargs(
            keys=self.data_buffer_objective_keys(),
            optimizer_config=self.optimizer_config,
        )
        super().optimize_tensor(**opt_kwargs)