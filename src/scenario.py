from regelum.utils import Clock, AwaitedParameter, calculate_value
from regelum.__internal.base import apply_callbacks

from regelum.scenario import RLScenario
from regelum.objective import RunningObjective
from regelum.critic import Critic

from regelum.simulator import  Simulator
from regelum.event import Event

from .objective import PortfolioRunningObjectiveModel, MarketRunningObjectiveModel
from .policy import JointPolicyVPG
import numpy as np
from copy import deepcopy

from typing import List
from typing_extensions import Self
import torch
import random

class GameScenario(RLScenario):
    def __init__(
        self,
        policy: JointPolicyVPG,
        portfolio_critic: Critic,
        market_critic: Critic,
        simulator: Simulator,
        portfolio_running_objective_model: PortfolioRunningObjectiveModel,
        market_running_objective_model: MarketRunningObjectiveModel,
        discount_factor: float = 0.95,
        sampling_time: float = 0.05,
        N_iterations: int = 200,
        iters_to_switch_opt_agent: int = 1,
        start_with_portfolio: bool = True
    ):
        self.portfolio_running_objective_model = portfolio_running_objective_model
        self.market_running_objective_model = market_running_objective_model
        self.portfolio_running_objective = RunningObjective(
            model=portfolio_running_objective_model
        )
        self.market_running_objective = RunningObjective(
            model=market_running_objective_model
        )
        self.portfolio_critic = portfolio_critic
        self.market_critic = market_critic

        self.iters_to_switch_opt_agent = iters_to_switch_opt_agent
        if start_with_portfolio:
            self.running_objective = self.portfolio_running_objective
            self.critic = portfolio_critic
        else:
            self.critic = market_critic
            self.running_objective = self.market_running_objective
        super().__init__(
            policy=policy,
            critic=self.critic,
            running_objective=self.running_objective,
            simulator=simulator,
            policy_optimization_event=Event.reset_iteration,
            critic_optimization_event=Event.reset_iteration,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            N_episodes=policy.N_episodes,
            N_iterations=N_iterations,
            is_critic_first=True
        )
        self.policy: JointPolicyVPG

    def instantiate_rl_scenarios(self):
        simulators = [deepcopy(self.simulator) for _ in range(self.N_episodes)]
        scenarios = [
            GameScenario(
                policy=self.policy,
                portfolio_critic=self.portfolio_critic,
                market_critic=self.market_critic,
                simulator=simulators[i],
                discount_factor=self.discount_factor,
                sampling_time=self.sampling_time,
                N_iterations=self.N_iterations,
                iters_to_switch_opt_agent = self.iters_to_switch_opt_agent,
                portfolio_running_objective_model = self.portfolio_running_objective_model,
                market_running_objective_model = self.market_running_objective_model,
                start_with_portfolio = self.running_objective is self.portfolio_running_objective
            )
            for i in range(self.N_episodes)
        ]
        return scenarios
    
    def switch_running_objective(self):
        self.running_objective = (
            self.portfolio_running_objective
            if self.running_objective is self.market_running_objective
            else self.market_running_objective
        )

    def switch_critic(self):
        self.critic = (
            self.portfolio_critic
            if self.critic is self.market_critic
            else self.market_critic
        )
    
    @apply_callbacks()
    def compute_action_sampled(self, time, estimated_state, observation):
        return super().compute_action_sampled(time, estimated_state, observation)

    @apply_callbacks()  # We will add a callbacks to log it when we switch an optimizing agent
    def switch_optimizing_agent(self):
        self.switch_running_objective()
        policy_weights_to_fix, policy_weights_to_unfix = (
            ["portfolio_model_weights", "market_model_weights"]
            if self.running_objective.model is self.market_running_objective.model
            else ["market_model_weights", "portfolio_model_weights"]
        )
        self.policy.fix_variables([policy_weights_to_fix])
        self.policy.unfix_variables([policy_weights_to_unfix])
        self.policy.switch_model_to_optimize()
        self.policy.switch_critic()
        self.switch_critic()
        return policy_weights_to_fix, policy_weights_to_unfix

    @apply_callbacks()
    def reset_iteration(self):
        super().reset_iteration()
        if self.iteration_counter % self.iters_to_switch_opt_agent == 0:
            self.switch_optimizing_agent()


    def on_action_issued(self, observation):
        action = self.get_action_from_policy()
        self.current_running_objective = self.running_objective(
            self.state, action
        )
        self.value = self.calculate_value(self.current_running_objective, self.time)
        observation_action = np.concatenate(
            (observation, action), axis=1
        )
        received_data =  {
            "action": action,
            "running_objective": self.current_running_objective,
            "current_value": self.value,
            "observation_action": observation_action,
            "state": self.state,
            "running_objective_market": self.market_running_objective(self.state, action),
            "running_objective_portfolio": self.portfolio_running_objective(self.state, action)
        }
        self.data_buffer.push_to_end(**received_data)
        return received_data


    @apply_callbacks()
    def post_compute_action(self, observation, estimated_state):
        action = self.get_action_from_policy()
        return {
            "estimated_state": estimated_state,
            "observation": observation,
            "time": self.time,
            "episode_id": self.episode_counter,
            "iteration_id": self.iteration_counter,
            "step_id": self.step_counter,
            "action": action,
            "running_objective": self.current_running_objective,
            "current_value": self.value,
            "state": self.state,
            "running_objective_market": self.market_running_objective(self.state, action),
            "running_objective_portfolio": self.portfolio_running_objective(self.state, action)
        }
    
    def run_ith_scenario(
        self, episode_id: int, iteration_id: int, scenarios: List[Self], queue
    ):
        seed = torch.initial_seed() + episode_id + iteration_id*self.policy.N_episodes
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed_increment += 1

        queue.put(
            (
                episode_id,
                scenarios[episode_id - 1].run_episode(episode_id, iteration_id),
            )
        )

    def run_episode(self, episode_counter, iteration_counter):
        self.episode_counter = episode_counter
        self.iteration_counter = iteration_counter
        while self.sim_status != "episode_ended":
            self.sim_status = self.step(episode_counter, iteration_counter)
        return self.data_buffer
    
    def step(self, episode_counter, iteration_counter):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if (not self.is_episode_ended) and (self.value <= self.value_threshold):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            if len(list(self.constraint_parser)) > 0:
                self.constraint_parameters = self.constraint_parser.parse_constraints(
                    simulation_metadata=self.simulation_metadata
                )
                self.substitute_constraint_parameters(**self.constraint_parameters)
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = self.simulator.do_sim_step(episode_counter, iteration_counter) == -1
            return "episode_continues"
        else:
            return "episode_ended"
