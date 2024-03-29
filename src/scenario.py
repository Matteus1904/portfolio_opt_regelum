
from regelum.__internal.base import apply_callbacks

from regelum.scenario import RLScenario
from regelum.objective import RunningObjective
from regelum.critic import Critic

from regelum.simulator import  Simulator
from regelum.event import Event

from .objective import PedestrianRunningObjectiveModel, ChauffeurRunningObjectiveModel
from .policy import JointPolicyVPG


class GameScenario(RLScenario):
    def __init__(
        self,
        policy: JointPolicyVPG,
        pedestrian_critic: Critic,
        chauffeur_critic: Critic,
        simulator: Simulator,
        pedestrian_running_objective_model: PedestrianRunningObjectiveModel,
        chauffeur_running_objective_model: ChauffeurRunningObjectiveModel,
        discount_factor: float = 0.95,
        sampling_time: float = 0.05,
        N_iterations: int = 200,
        iters_to_switch_opt_agent: int = 1,
    ):
        self.pedestrian_running_objective = RunningObjective(
            model=pedestrian_running_objective_model
        )
        self.chauffeur_running_objective = RunningObjective(
            model=chauffeur_running_objective_model
        )
        self.pedestrian_critic = pedestrian_critic
        self.chauffeur_critic = chauffeur_critic

        self.iters_to_switch_opt_agent = iters_to_switch_opt_agent
        super().__init__(
            policy=policy,
            critic=pedestrian_critic,
            running_objective=self.pedestrian_running_objective,
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
            self.pedestrian_running_objective
            if self.running_objective is self.chauffeur_running_objective
            else self.chauffeur_running_objective
        )

    def switch_critic(self):
        self.critic = (
            self.pedestrian_critic
            if self.critic is self.chauffeur_critic
            else self.chauffeur_critic
        )

    @apply_callbacks()
    def compute_action_sampled(self, time, estimated_state, observation):
        return super().compute_action_sampled(time, estimated_state, observation)

    @apply_callbacks()  # We will add a callbacks to log it when we switch an optimizing agent
    def switch_optimizing_agent(self):
        self.switch_running_objective()
        policy_weights_to_fix, policy_weights_to_unfix = (
            ["pedestrian_model_weights", "chauffeur_model_weights"]
            if self.running_objective.model is self.chauffeur_running_objective.model
            else ["chauffeur_model_weights", "pedestrian_model_weights"]
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