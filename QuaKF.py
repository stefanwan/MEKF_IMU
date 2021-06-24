from math import sin, cos, atan, atan2, asin, sqrt
import numpy as np
from KalmanFilter import KF

class QuaternionKF(KF):

    def __init__(self, 
                 step_time, 
                 dimension_of_state, 
                 dimension_of_ob=None, 
                 variance_of_model=.5, 
                 variance_of_measurement=.001, 
                 covariance_init=1e6):
        super().__init__(step_time, dimension_of_state, dimension_of_ob, 
                         variance_of_model, variance_of_measurement, covariance_init)
        self.reset(covariance_init, np.zeros((3, 1)))

    def reset(self, covariance, x):
        self.p = np.identity(self.sdim) * covariance
        self.q_est = self.Euler2Quaternion(x)

        return self.q_est

    def predict(self, u):
        u = u/180*np.pi
        dq = self.ExpQua(u*self.dt)
        self.q_est = self.MulQua(self.q_est, dq)
        Phi = self.cal_Phi(dq)
        G = self.cal_G()
        self.p = Phi @ self.p @ Phi.T + G @ self.Q * self.dt**2 @ G.T
    
    def update(self, observation):
        accx = observation[0]
        accy = observation[1]
        accz = observation[2]
        K = self.p @ np.linalg.inv(self.p + self.R)
        angle_est = self.Quaternion2Euler(self.q_est)
        z = np.array([[atan2(accy, accz)],
                      [atan(-accx/sqrt(accy**2+accz**2))],
                      [angle_est[2]]], dtype=np.float)
        qmea = self.Euler2Quaternion(z)
        dq = qmea - self.q_est
        self.q_est = self.q_est + K @ dq
        self.q_est = self.q_est / np.linalg.norm(self.q_est, ord=2)
        self.p = (np.identity(self.sdim) - K) @ self.p

        return self.q_est

    def ExpQua(self, v):
        q = np.zeros([4], dtype=np.float)
        v = v / 2
        theta = np.linalg.norm(v, ord=2)
        if theta < 1e-4:
            q = np.array([1, 0, 0, 0])
        else:
            q[0] = cos(theta)
            q[1:] = v/theta*sin(theta)

        return q

    def MulQua(self, q1, q2):
        q = np.zeros([4], dtype=np.float64)

        q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

        return q

    def Euler2Quaternion(self, euler_angle):
        q = np.zeros([4], dtype=np.float)
        
        q[0] = cos(euler_angle[0]/2)*cos(euler_angle[1]/2)*cos(euler_angle[2]/2) + \
                sin(euler_angle[0]/2)*sin(euler_angle[1]/2)*sin(euler_angle[2]/2)
        q[1] = sin(euler_angle[0]/2)*cos(euler_angle[1]/2)*cos(euler_angle[2]/2) - \
                cos(euler_angle[0]/2)*sin(euler_angle[1]/2)*sin(euler_angle[2]/2)
        q[2] = cos(euler_angle[0]/2)*sin(euler_angle[1]/2)*cos(euler_angle[2]/2) + \
                sin(euler_angle[0]/2)*cos(euler_angle[1]/2)*sin(euler_angle[2]/2)
        q[3] = cos(euler_angle[0]/2)*cos(euler_angle[1]/2)*sin(euler_angle[2]/2) - \
                sin(euler_angle[0]/2)*sin(euler_angle[1]/2)*cos(euler_angle[2]/2)

        return q

    def Quaternion2Euler(self, q):
        euler_angle = np.zeros([3], dtype=np.float64)

        euler_angle[0] = atan2(2.0*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2)
        euler_angle[1] = -asin(np.clip(2.0*(q[1]*q[3]-q[0]*q[2]), -1.0, 1.0))
        euler_angle[2] = atan2(2*(q[1]*q[2]+q[0]*q[3]), q[0]**2+q[1]**2-q[2]**2-q[3]**2)

        return euler_angle

    def cal_Phi(self, dq):
        Phi = np.array([[dq[0], -dq[1], -dq[2], -dq[3]],
                        [dq[1], dq[0], dq[3], -dq[2]],
                        [dq[2], -dq[3], dq[0], dq[1]],
                        [dq[3], dq[2], -dq[1], dq[0]]], dtype=np.float)

        return Phi

    def cal_G(self):
        G = np.array([[self.q_est[0], -self.q_est[1], -self.q_est[2], -self.q_est[3]],
                      [self.q_est[1], self.q_est[0], -self.q_est[3], self.q_est[2]],
                      [self.q_est[2], self.q_est[3], self.q_est[0], -self.q_est[1]],
                      [self.q_est[3], -self.q_est[2], self.q_est[1], self.q_est[0]]], dtype=np.float)

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

    QuaKF = QuaternionKF(dimension_of_state=4, step_time=0.01)
    angle_filter = []
    for i in range(omega_base.shape[0]):
        angle_filter.append(QuaKF.Quaternion2Euler(QuaKF.step(omega_base[i, :], acc_base[i, :])))

    angle_filter = np.asarray(angle_filter)

    plt.figure()
    plt.suptitle("Altitude Estimation using QuatenrionKF")
    plt.subplot(3, 1, 1)
    plt.ylabel('Euler Angle X-Axis')
    plt.plot(angle_filter[:, 0]/np.pi*180, '-')
    plt.plot(angle_base[:, 0], '--')
    plt.legend(['filter', 'base'])

    plt.subplot(3, 1, 2)
    plt.ylabel('Euler Angle Y-Axis')
    plt.plot(angle_filter[:, 1]/np.pi*180, '-')
    plt.plot(angle_base[:, 1], '--')
    plt.legend(['filter', 'base'])

    plt.subplot(3, 1, 3)
    plt.xlabel('Frame Number')
    plt.ylabel('Euler Angle Z-Axis')
    plt.plot(angle_filter[:, 2]/np.pi*180, '-')
    plt.plot(angle_base[:, 2], '--')
    plt.legend(['filter', 'base'])
    plt.show()