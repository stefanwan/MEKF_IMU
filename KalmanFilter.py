import numpy as np
from abc import ABC, abstractmethod

class KF(ABC):
    """ 
    An abstract Kalman Filter algorithm.
    Used to provide fusion estimation of attitude, taking 
    angular velocity and acceleration which sampling from IMU as inputs. 
    """
    def __init__(self, 
                 step_time, 
                 dimension_of_state,
                 dimension_of_ob, 
                 variance_of_model, 
                 variance_of_measurement,
                 covariance_init):
        self.sdim = dimension_of_state
        self.odim = dimension_of_ob or dimension_of_state
        self.dt = step_time
        self.Q = np.identity(self.sdim) * variance_of_model
        self.R = np.identity(self.odim) * variance_of_measurement

    @abstractmethod  # Subclass must re-write this method or an error will be reported
    def reset(self, covariance, x):
        """ 
        Reset the initial values of covariance matrix and estimated state 
        with given covariance and x.
        """

        pass

    @abstractmethod
    def predict(self, u):
        """ 
        Take one step prediction using angular velocity and system model, and
        calculate the covariance matirx using current info. 
        """
        
        pass

    @abstractmethod
    def update(self, observation):
        """ 
        Update the estimation when observation info comes.
        The observation info is acceleration values and will be solved to get attitude. 
        Finally, update the covariance matirx. 
        """

        raise NotImplementedError

    def step(self, u, observation):
        """ 
        Step the filter algorithm.
        Through one step predicting and one step updating, return
        the altitude estimation 
        """
        self.predict(u)
        
        return self.update(observation)

