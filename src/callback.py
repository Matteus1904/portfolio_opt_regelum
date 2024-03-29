from regelum.callback import (
    Callback,
    HistoricalCallback,
    ScenarioStepLogger,
    HistoricalDataCallback,
    ValueCallback
)
from .scenario import GameScenario
from .policy import JointPolicyVPG

class SwitchAgentCallback(Callback):
    def is_target_event(self, obj, method, output, triggers):
        return method == "switch_optimizing_agent"

    def on_function_call(self, obj: GameScenario, method, output):
        self.log(
            "Optimizing agent switched to"
            f" {'portfolio' if output[1] == 'portfolio_model_weights' else 'market'} "
            f"after iterartion: {obj.iteration_counter}"
        )

class DoubleAgentStepLogger(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cooldown = 0.0
        self.state_components_naming = ['cash [USD]'] + [f'volume_{i}' for i in range(4)] + [f'price_{i} [USD]' for i in range(4)]

    def is_target_event(self, obj, method, output, triggers):
        return (method == "post_compute_action") and isinstance(obj, GameScenario)

    def on_function_call(self, obj: GameScenario, method, output):
        self.add_datum(
            {
                **{
                    "time": output["time"],
                    "running_objective_portfolio": obj.portfolio_running_objective(
                        output["estimated_state"], output["action"]
                    ),
                    "running_objective_market": obj.market_running_objective(
                        output["estimated_state"], output["action"]
                    ),
                    "episode_id": output["episode_id"],
                    "iteration_id": output["iteration_id"],
                },
                **dict(zip(self.state_components_naming, output["estimated_state"][0])),
            }
        )


class WhichOptimizeCallback(Callback):
    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, JointPolicyVPG) and method == "optimize"

    def on_function_call(self, obj: JointPolicyVPG, method, output):
        which = (
            "portfolio"
            if obj.market_model_weights.is_constant
            else "market_model"
        )
        self.log(f"A {which} policy updated...")