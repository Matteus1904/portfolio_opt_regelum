from regelum.callback import (
    Callback,
    HistoricalCallback,
    HistoricalDataCallback
)
from .scenario import GameScenario
from .policy import JointPolicyVPG
import pandas as pd
import numpy as np


class HistoricalDataCallback(HistoricalDataCallback):
    def on_function_call(self, obj, method, output):
        if self.observation_components_naming is None:
            self.observation_components_naming = (
                [
                    f"observation_{i + 1}"
                    for i in range(obj.simulator.system.dim_observation)
                ]
                if obj.simulator.system.observation_naming is None
                else obj.simulator.system.observation_naming
            )

        if self.action_components_naming is None:
            self.action_components_naming = (
                [f"action_{i + 1}" for i in range(obj.simulator.system.dim_inputs)]
                if obj.simulator.system.inputs_naming is None
                else obj.simulator.system.inputs_naming
            )

        if self.state_components_naming is None:
            self.state_components_naming = (
                [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
                if obj.simulator.system.state_naming is None
                else obj.simulator.system.state_naming
            )

        if method == "post_compute_action":
            self.add_datum(
                {
                    **{
                        "time": output["time"],
                        "current_value": output["current_value"],
                        "episode_id": output["episode_id"],
                        "iteration_id": output["iteration_id"],
                        "running_objective_portfolio": output["running_objective_portfolio"],
                        "running_objective_market": output["running_objective_market"],
                    },
                    **dict(zip(self.action_components_naming, output["action"][0])),
                    **dict(
                        zip(self.observation_components_naming, output["observation"][0])
                    ),
                    **dict(
                        zip(self.state_components_naming, output["state"][0])
                    ),
                }
            )
        elif method == "dump_data_buffer":
            _, data_buffer = output
            self.data = pd.concat(
                [
                    data_buffer.to_pandas(
                        keys={
                            "time": float,
                            "current_value": float,
                            "episode_id": int,
                            "iteration_id": int,
                            "running_objective_portfolio": float,
                            "running_objective_market": float

                        }
                    )
                ]
                + [
                    pd.DataFrame(
                        columns=columns,
                        data=np.array(
                            data_buffer.to_pandas([key]).values.tolist(),
                            dtype=float,
                        ).squeeze(),
                    )
                    for columns, key in [
                        (self.action_components_naming, "action"),
                        (self.observation_components_naming, "observation"),
                        (self.state_components_naming, "state"),
                    ]
                ],
                axis=1,
            )


    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        if episodes_total == 1:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}"
        else:
            identifier = f"observations_actions_it_{str(iteration_number).zfill(5)}_ep_{str(episode_number).zfill(5)}"
        self.dump_and_clear_data(identifier)

    def plot(self, name=None):
        pass

class SwitchAgentCallback(Callback):
    def is_target_event(self, obj, method, output, triggers):
        return method == "switch_optimizing_agent"

    def on_function_call(self, obj: GameScenario, method, output):
        self.log(
            "Optimizing agent switched to"
            f" {'portfolio' if output[1] == 'portfolio_model_weights' else 'market'} "
            f"after iterartion: {obj.iteration_counter}"
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