from regelum.simulator import Simulator
from regelum.utils import rg

class Simulator(Simulator):
    def do_sim_step(self):
        """Do one simulation step and update current simulation data (time, system state and output)."""
        if self.system.system_type == "discrete_stoch":
            if self.time <= self.time_final:
                self.state_init = self.state_init.copy()
                self.system._step_size = self.max_step
                self.state += (
                    self.system.compute_state_dynamics(time = None, state = rg.array(self.state), inputs=rg.array(self.system.inputs))
                )
                self.time += self.max_step
                self.observation = self.state
            else:
                self.reset()
                return -1
        else: 
            raise ValueError("Invalid system description")
        
    def reset(self):
        if self.system.system_type == "discrete_stoch":
            self.time = 0.0
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
        else: 
            raise ValueError("Invalid system description")