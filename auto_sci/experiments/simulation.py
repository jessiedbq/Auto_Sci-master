"""
File: simulation.py
Date: May 12 2019
Author: Jessie Deng

This module implements the simulation of dynamical system by numerical
integration.
A few rules that MUST be followed:
    1. TODO
    2. TODO
"""

# Libraries
from scipy.integrate import odeint
import numpy as np
import os
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Union, Tuple
from scipy.integrate import ode




class DynamicalSystem(ABC):
    """
    Abstract class for a dynamical system. Includes an ODE solver based on
    scipy.
    """

    def __init__(self, dimensionality: int,
                 true_param: Union[list, np.array],
                 noise_variance: float = 0.0,
                 stn_ratio: float = None):
        """
        General Constructor.
        :param dimensionality: dimension of the state of the system;
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        self.dim = dimensionality
        self.theta = np.array(true_param)
        self.mean = 0.0
        self.variance = noise_variance
        self.system_ode = ode(self._system_ode).set_integrator('vode',
                                                               method='bdf')
        self.stn_ratio = stn_ratio
        return

    @staticmethod
    @abstractmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        return []

    def simulate(self, initial_state: Union[list, np.array],
                 initial_time: float, final_time: float,
                 t_delta_integration: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system = np.copy(initial_state).reshape(self.dim, 1)
        t = [initial_time]
        self.system_ode.set_initial_value(initial_state,
                                          initial_time).set_f_params(self.theta)
        while self.system_ode.successful() and self.system_ode.t < final_time:
            self.system_ode.integrate(self.system_ode.t + t_delta_integration)
            system = np.c_[system, self.system_ode.y.reshape(self.dim, 1)]
            t.append(self.system_ode.t)
        return system, np.array(t)

    def observe(self, initial_state: Union[list, np.array],
                initial_time: float, final_time: float,
                t_delta_integration: float,
                t_delta_observation: float) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        [system, t] = self.simulate(initial_state,
                                    initial_time,
                                    final_time,
                                    t_delta_integration)
        t_obs = np.arange(initial_time, final_time + t_delta_observation,
                          t_delta_observation)
        observed_system = np.zeros([self.dim, t_obs.shape[0]])
        for n in range(self.dim):
            observed_system[n, :] = np.interp(t_obs, t, system[n, :])
        if self.variance != 0.0:
            noise = np.random.normal(loc=0.0, scale=np.sqrt(self.variance),
                                     size=observed_system.shape)
            observed_system += noise.reshape(observed_system.shape)
        if self.stn_ratio:
            std_devs_signal = np.std(observed_system, axis=0)
            std_devs_noise = std_devs_signal / np.sqrt(self.stn_ratio)
            noise = np.random.normal(loc=0.0, scale=1.0,
                                     size=observed_system.shape)
            for n in range(self.dim):
                noise[n, :] = noise[n, :] * std_devs_noise[n]
            observed_system += noise.reshape(observed_system.shape)
        return observed_system, t_obs.reshape(-1, 1)

    def observe_at_t(self, initial_state: Union[list, np.array],
                     initial_time: float, final_time: float,
                     t_delta_integration: float,
                     t_observations: np.array):
        """"
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations, computed at the time stamps specified in
        t_observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_observations: time stamps at which observe the system.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        [system, t] = self.simulate(initial_state,
                                    initial_time,
                                    final_time,
                                    t_delta_integration)
        t_obs = t_observations
        observed_system = np.zeros([self.dim, t_obs.shape[0]])
        for n in range(self.dim):
            observed_system[n, :] = np.interp(t_obs, t, system[n, :])
        if self.variance != 0.0:
            noise = np.random.normal(loc=0.0, scale=np.sqrt(self.variance),
                                     size=observed_system.shape)
            observed_system += noise.reshape(observed_system.shape)
        if self.stn_ratio:
            std_devs_signal = np.std(observed_system, axis=0)
            std_devs_noise = std_devs_signal / np.sqrt(self.stn_ratio)
            noise = np.random.normal(loc=0.0, scale=1.0,
                                     size=observed_system.shape)
            for n in range(self.dim):
                noise[n, :] = noise[n, :] * std_devs_noise[n]
            observed_system += noise.reshape(observed_system.shape)
        return observed_system, t_obs.reshape(-1, 1)


class LotkaVolterra(DynamicalSystem):
    """
    2D Lotka-Volterra ODE.
    """

    def __init__(self,
                 true_param: Union[list, np.array] = (2.0, 1.0, 4.0, 1.0),
                 noise_variance: float = 0.1 ** 2,
                 stn_ratio: float = None):
        """
        Constructor.
        :param true_param: true parameters of the system;
        :param noise_variance: variance of the observation noise, if different
        from zero it overwrites the signal to noise ratio;
        :param stn_ratio: signal to noise ratio (variance should be set to zero
        if the stn_ratio is different than None).
        """
        super(LotkaVolterra, self).__init__(2,
                                            true_param,
                                            noise_variance,
                                            stn_ratio)
        assert self.theta.shape[0] == 4,\
            "Error: length of true_param should be 4"
        return

    @staticmethod
    def _system_ode(t: float, y: np.array,
                    theta: np.array) -> list:
        """
        Describes the overall evolution of the system in the form:
                dy / dt = f( t, y, args)
        Needed by scipy.
        :param t: time, needed in arguments even if it's not directly used;
        :param y: current state;
        :param theta: arguments and parameters of the system.
        :return: the f function so built.
        """
        f = [theta[0] * y[0] - theta[1] * y[0] * y[1],
             - theta[2] * y[1] + theta[3] * y[0] * y[1]]
        return f

    def simulate(self,
                 initial_state: Union[list, np.array] = (5.0, 3.0),
                 initial_time: float = 0.0,
                 final_time: float = 2.0,
                 t_delta_integration: float = 0.01)\
            -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals.
        :return: a numpy array containing the integrated dynamical system (of
        size [n_states, n_points]) and a numpy array containing the time stamps.
        """
        system, t = super(LotkaVolterra, self).simulate(initial_state,
                                                        initial_time,
                                                        final_time,
                                                        t_delta_integration)
        return system, t

    def observe(self, initial_state: Union[list, np.array] = (5.0, 3.0),
                initial_time: float = 0.0,
                final_time: float = 2.0,
                t_delta_integration: float = 0.01,
                t_delta_observation: float = 0.1) -> Tuple[np.array, np.array]:
        """
        Integrate the system using an scipy built-in ODE solver and extract the
        noisy observations.
        :param initial_state: initial state of the system;
        :param initial_time: initial time of the simulation;
        :param final_time: final time of the simulation;
        :param t_delta_integration: time between integration intervals;
        :param t_delta_observation: time between observation intervals.
        :return: a numpy array containing the noisy observations of the
        integrated dynamical system (of size [n_states, n_points])
        and a numpy array containing the time stamps.
        """
        observed_system, t = super(LotkaVolterra,
                                   self).observe(initial_state,
                                                 initial_time,
                                                 final_time,
                                                 t_delta_integration,
                                                 t_delta_observation)
        return observed_system, t
