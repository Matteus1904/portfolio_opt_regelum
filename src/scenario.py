
from regelum.__internal.base import apply_callbacks

from regelum.scenario import RLScenario
from regelum.objective import RunningObjective
from regelum.critic import Critic

from regelum.simulator import  Simulator
from regelum.event import Event

from .objective import PortfolioRunningObjectiveModel, MarketRunningObjectiveModel
from .policy import JointPolicyVPG


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
    ):
        self.portfolio_running_objective = RunningObjective(
            model=portfolio_running_objective_model
        )
        self.market_running_objective = RunningObjective(
            model=market_running_objective_model
        )
        self.portfolio_critic = portfolio_critic
        self.market_critic = market_critic

        self.iters_to_switch_opt_agent = iters_to_switch_opt_agent
        super().__init__(
            policy=policy,
            critic=portfolio_critic,
            running_objective=self.portfolio_running_objective,
            simulator=simulator,
            policy_optimization_event=Event.reset_iteration,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            N_episodes=policy.N_episodes,
            N_iterations=N_iterations,
            is_critic_first=True,
        )
        self.policy: JointPolicyVPG

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

    def reset_iteration(self):
        super().reset_iteration()
        if self.iteration_counter % self.iters_to_switch_opt_agent == 0:
            self.switch_optimizing_agent()