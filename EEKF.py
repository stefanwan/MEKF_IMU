import numpy as np
from math import sin, cos, tan, atan2, atan, sqrt
from KalmanFilter import KF

class EulerEKF(KF):
    
    Gamma = np.identity(6)
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    def __init__(self, 
                 step_time,
                 dimension_of_state, 
                 dimension_of_ob=None,
                 variance_of_model=.5, 
                 variance_of_measurement=.001, 
                 covariance_init=1e6):
        super().__init__(step_time, dimension_of_state,  dimension_of_ob,
                         variance_of_model, variance_of_measurement, covariance_init)
        self.reset(covariance_init, np.zeros((self.sdim, 1)))

    def reset(self, covariance, x):
        self.p = np.identity(self.sdim) * covariance
        self.x_est = x

        return self.x_est

    def predict(self, u):
        u = u/180*np.pi
        Phi = self.cal_Phi(u)
        G = self.cal_G()
        u = u[:, np.newaxis]
        self.x_est = Phi @ self.x_est + G @ u
        self.p = Phi @ self.p @ Phi.T + self.Gamma @ self.Q * self.dt**2  @ self.Gamma.T

    def update(self, observation):
        accx = observation[0]
        accy = observation[1]
        accz = observation[2]
        K = self.p @ self.H.T @ np.linalg.inv(self.H @ self.p @ self.H.T + self.R)
        z = np.array([[atan2(accy, accz)],
                      [atan(-accx/sqrt(accy**2+accz**2))],
                      [0]])
        self.x_est = self.x_est + K @ (z - self.H @ self.x_est)
        self.p = (np.identity(self.sdim) - K @ self.H) @ self.p

        return self.x_est

    def cal_Phi(self, u):
        gy = u[1]
        gz = u[2]
        phi_11 = 1 + ((gy-self.x_est[4, 0])*cos(self.x_est[0, 0])*tan(self.x_est[1, 0]) - 
                 (gz-self.x_est[5, 0])*sin(self.x_est[0, 0])*tan(self.x_est[1, 0]))*self.dt
        phi_12 = ((gy-self.x_est[4, 0])*sin(self.x_est[0, 0]) + 
                 (gz-self.x_est[5, 0])*cos(self.x_est[0, 0]))/(cos(self.x_est[1, 0])**2+1e-4)*self.dt
        phi_13 = 0
        phi_14 = -self.dt
        phi_15 = -sin(self.x_est[0, 0])*tan(self.x_est[1, 0])*self.dt
        phi_16 = -cos(self.x_est[0, 0])*tan(self.x_est[1, 0])*self.dt
        phi_21 = (-(gy-self.x_est[4, 0])*sin(self.x_est[0, 0]) - 
                 (gz-self.x_est[5, 0])*cos(self.x_est[0, 0]))*self.dt
        phi_22 = 1
        phi_23 = 0
        phi_24 = 0
        phi_25 = -cos(self.x_est[0, 0])*self.dt
        phi_26 = sin(self.x_est[0, 0])*self.dt
        phi_31 = ((gy-self.x_est[4, 0])*cos(self.x_est[0, 0]) - 
                 (gz-self.x_est[5, 0])*sin(self.x_est[0, 0]))/(cos(self.x_est[1, 0])+1e-4)*self.dt
        phi_32 = ((gy-self.x_est[4, 0])*sin(self.x_est[0, 0])*sin(self.x_est[1, 0]) + 
                 (gz-self.x_est[5, 0])*cos(self.x_est[0, 0])*sin(self.x_est[1, 0]))/(cos(self.x_est[1, 0])**2+1e-4)*self.dt
        phi_33 = 1
        phi_34 = 0
        phi_35 = -sin(self.x_est[0, 0])/(cos(self.x_est[1, 0])+1e-4)*self.dt
        phi_36 = -cos(self.x_est[0, 0])/(cos(self.x_est[1, 0])+1e-4)*self.dt
        phi_41 = phi_42 = phi_43 = phi_45 = phi_46 = 0
        phi_44 = 1
        phi_51 = phi_52 = phi_53 = phi_54 = phi_56 = 0
        phi_55 = 1
        phi_61 = phi_62 = phi_63 = phi_64 = phi_65 = 0
        phi_66 = 1
        Phi = np.array([[phi_11, phi_12, phi_13, phi_14, phi_15, phi_16],
                        [phi_21, phi_22, phi_23, phi_24, phi_25, phi_26],
                        [phi_31, phi_32, phi_33, phi_34, phi_35, phi_36],
                        [phi_41, phi_42, phi_43, phi_44, phi_45, phi_46],
                        [phi_51, phi_52, phi_53, phi_54, phi_55, phi_56],
                        [phi_61, phi_62, phi_63, phi_64, phi_65, phi_66]])

        return Phi

    def cal_G(self):
        g_11 = self.dt
        g_12 = sin(self.x_est[0, 0])*tan(self.x_est[1, 0])*self.dt
        g_13 = cos(self.x_est[0, 0])*tan(self.x_est[1, 0])*self.dt
        g_21 = 0
        g_22 = cos(self.x_est[0 ,0])*self.dt
        g_23 = -sin(self.x_est[0, 0])*self.dt
        g_31 = 0
        g_32 = sin(self.x_est[0, 0])/(cos(self.x_est[1, 0]) + 1e-4)*self.dt
        g_33 = cos(self.x_est[0, 0])/(cos(self.x_est[1, 0]) + 1e-4)*self.dt
        g_41 = g_42 = g_43 = g_51 = g_52 = g_53 = g_61 = g_62 = g_63 = 0
        G = np.array([[g_11, g_12, g_13],
                      [g_21, g_22, g_23],
                      [g_31, g_32, g_33],
                      [g_41, g_42, g_43],
                      [g_51, g_52, g_53],
                      [g_61, g_62, g_63]])

        return G


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    usecols = ['ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)', 'AngleX(deg)',
           'AngleY(deg)', 'AngleZ(deg)']
    data_base = pd.read_csv('../data/201217095118.txt', usecols=usecols ,sep=r'\s+')
    acc_base = data_base.loc[:, ['ax(g)', 'ay(g)', 'az(g)']].values
    omega_base = data_base.loc[:, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']].values
    angle_base = data_base.loc[:, ['AngleX(deg)', 'AngleY(deg)', 'AngleZ(deg)']].values

    EEKF = EulerEKF(dimension_of_state=6, dimension_of_ob=3, step_time=0.01)
    angle_filter = []
    for i in range(omega_base.shape[0]):
        x_est = EEKF.step(omega_base[i, :], acc_base[i, :])
        angle_filter.append(np.squeeze(x_est[:3]))

    angle_filter = np.asarray(angle_filter)
    plt.figure()
    plt.plot(angle_filter[:, 0]/np.pi*180)
    plt.plot(angle_base[:, 0])
    plt.legend(['filter', 'base'])
    plt.figure()
    plt.plot(angle_filter[:, 1]/np.pi*180)
    plt.plot(angle_base[:, 1])
    plt.legend(['filter', 'base'])

    plt.figure()
    plt.plot(angle_filter[:, 2]/np.pi*180)
    plt.plot(angle_base[:, 2])
    plt.legend(['filter', 'base'])
    plt.show()

