from regelum.simulator import Simulator
from regelum.utils import rg
from regelum.system import System, ComposedSystem
from typing import Union, Optional

import requests
import pandas as pd
import math
from datetime import datetime
import numpy as np
import random

class Simulator(Simulator):
    def do_sim_step(self):
        """Do one simulation step and update current simulation data (time, system state and output)."""
        if self.system.system_type == "discrete_stoch":
            if self.time <= self.time_final:
                self.state_init = self.state_init.copy()
                self.system._step_size = self.max_step
                self.state += (
                    self.system.compute_state_dynamics(time = None, state = self.state, inputs=self.system.inputs)
                )
                self.observation = self.get_observation(
                    time=self.time, state=self.state, inputs=self.system.inputs
                )
                self.time += self.max_step
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
        

class Historical_Simulator(Simulator):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Optional[np.ndarray] = None,
        action_init: Optional[np.ndarray] = None,
        time_final: Optional[float] = 1,
        max_step: Optional[float] = 1e-3,
        first_step: Optional[float] = 1e-6,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
    ):
        self.system = system

        self.time_final = time_final
        if state_init is None:
            self.state_init = self.initialize_init_state()
        else:
            self.state_init = state_init
        if action_init is None:
            self.action_init = self.initialize_init_action()
        else:
            self.action_init = action_init

        self.time = 0.0
        self.state = self.state_init
        self.observation_init = self.observation = self.get_observation(
            time=self.time, state=self.state_init, inputs=self.action_init
        )

        self.max_step = max_step
        self.atol = atol
        self.rtol = rtol
        self.first_step = first_step
        self.number_of_steps = int(self.time_final/self.max_step)
        self.prices = np.concatenate([klines(market = 'AAVEUSDT'), klines(market = 'SOLUSDT'), klines(market = 'LTCUSDT')], axis = 1)
        self.number_of_batches = self.prices.shape[0]//self.number_of_steps

    def do_sim_step(self):
        """Do one simulation step and update current simulation data (time, system state and output)."""
        number_of_stocks = self.prices.shape[1]
        current_step = int(self.time/self.max_step)
        if self.time == 0.0:
            self.current_batch = random.randint(0, self.number_of_batches-1)
            self.state[:, 4+2*number_of_stocks: 4+3*number_of_stocks] = self.prices[self.current_batch*self.number_of_steps + current_step]
            self.state[:, 4+3*number_of_stocks: ] = self.prices[self.current_batch*self.number_of_steps + current_step]
        if self.system.system_type == "discrete_stoch":
            if self.time <= self.time_final:
                self.state_init = self.state_init.copy()
                self.system._step_size = self.max_step
                self.state += (
                    self.system.compute_state_dynamics(time = None, state = self.state, inputs=self.system.inputs)
                )
                self.state[:, 4+2*number_of_stocks: 4+3*number_of_stocks] = self.prices[self.current_batch*self.number_of_steps + current_step + 1]
                self.state[:, 4+3*number_of_stocks: ] = self.prices[self.current_batch*self.number_of_steps + current_step]
                self.observation = self.get_observation(
                    time=self.time, state=self.state, inputs=self.system.inputs
                )
                self.time += self.max_step
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
        

def klines(market = 'BTCUSDT', tick_interval = '30m', startDay = "03/10/2023", endDay="03/04/2024"):
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    def convert_to_seconds(s):
        return int(s[:-1]) * seconds_per_unit[s[-1]]
    startTime = int(datetime.strptime(startDay,"%d/%m/%Y").timestamp() + 10800 - convert_to_seconds(tick_interval))
    endTime = int(datetime.strptime(endDay,"%d/%m/%Y").timestamp() + 10799)
    interval = convert_to_seconds(tick_interval)
    n = (endTime - startTime)/interval
    for i in range(math.ceil(n/1000)):
        start = startTime +i*interval*1000
        end = min(start+interval*1000, endTime)
        url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval+'&startTime='+ str(start*1000)+'&endTime='+str(end*1000)+'&limit=1000'
        data_one = pd.DataFrame(requests.get(url).json())[[0, 4]]
        data_one.rename(columns={0: 'timestamp', 4: 'close_price'}, inplace=True)
        data_one.sort_values(by=['timestamp'], inplace=True)
        if i==0:
            data = data_one
        else:
            data = pd.concat([data, data_one], ignore_index=True)
    data['timestamp'] = pd.to_datetime(data['timestamp']/1000, unit='s')
    data['close_price'] = pd.to_numeric(data['close_price'])
    data['returns'] = (data.close_price -data.close_price.shift(1))/ data.close_price.shift(1)
    data = data.iloc[1:].reset_index(drop = True)
    data = np.array(data[['close_price']])
    return data


prices = np.concatenate([klines(market = 'BTCUSDT'), klines(market = 'ETHUSDT'), klines(market = 'SOLUSDT')], axis = 1)