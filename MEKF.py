import math
import numpy as np
from KalmanFilter import KF

class MultiplicativeEKF(KF):

    def __init__(self, 
                 step_time, 
                 dimension_of_state, 
                 dimension_of_ob=None,
                 variance_of_model=.5, 
                 variance_of_measurement=.1,
                 covariance_init=1e6):
        KF.__init__(self, step_time, dimension_of_state, dimension_of_ob,  
                    variance_of_model, variance_of_measurement, covariance_init)
        self.reset(covariance_init, np.zeros((self.sdim, 1)))
        
    def reset(self, covariance, x):
        self.p = np.identity(self.sdim) * covariance
        self.q_est = self.Euler2Quaternion(x)

        return self.q_est

    def predict(self, u):
        u = u/180*math.pi
        dq = self.ExpQua(u*self.dt)
        self.q_est = self.MulQua(self.q_est, dq)
        Phi = self.ExpRot(u*self.dt)
        self.p = Phi.T @ self.p @ Phi + self.Q * self.dt**2

    def update(self, observation):
        K = self.p @ np.linalg.inv(self.p + self.R)
        rotation_matrix = self.Quaternion2RotationMatrix(self.q_est)
        acc_est = np.squeeze(rotation_matrix @ np.array([[0], [0], [1]]))
        acc_mea = observation / (np.linalg.norm(observation, ord=2)+1e-5)
        acc_err = np.cross(acc_mea, acc_est)
        if np.dot(acc_est, acc_mea) < 0:
            mode = np.linalg.norm(acc_err, ord=2)
            acc_err = (2 - mode) * acc_err
        dv = K @ acc_err 
        self.q_est = self.MulQua(self.q_est, self.ExpQua(dv))
        self.p = (np.identity(self.sdim) - K) @ self.p

        return self.q_est

    def _WraptoPi(self, x):
        xwrap = x % (2*np.pi)
        if abs(xwrap) > np.pi:
            xwrap = xwrap - 2*np.pi*np.sign(xwrap)

        return xwrap

    def ExpQua(self, v):
        q = np.zeros([4], dtype=np.float64)
        v = v / 2
        theta = np.linalg.norm(v, ord=2)
        if theta < 1e-4:
            q = np.array([1, 0, 0, 0])
        else:
            q[0] = math.cos(theta)
            q[1:] = v/theta*math.sin(theta)

        return q

    def LogQua(self, q):
        norm_qv = np.linalg.norm(q[1:], ord=2)
        if abs(q[0]-1) <= 1e-4:
            u = np.array([[1],
                          [0],
                          [0]])
            theta = 0
        else:
            u = q[1:]/norm_qv
            theta = self._WraptoPi(math.atan2(norm_qv, q[0])*2)
        v = u*theta

        return v

    def InvQua(self, q):
        q_inv = np.zeros([4], dtype=np.float64)

        q_inv[0] = q[0]
        q_inv[1:4] = -q[1:4]

        return q_inv

    def MulQua(self, q1, q2):
        q = np.zeros([4], dtype=np.float64)

        q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

        return q

    def ExpRot(self, v):
        normv = np.linalg.norm(v, ord=2)

        s = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]], dtype=np.float64)
        if normv < 1e-4:
            RotM = np.identity(3)
        else:
            RotM = np.identity(3) + math.sin(normv)/normv*s + (1-math.cos(normv))/normv**2*(s@s)
        
        return RotM
    
    def Quaternion2RotationMatrix(self, q):
        q_scalar = q[0]
        q_vector = q[1:4][:, np.newaxis]
        q_skew = np.array([[0, -self.q_est[3], self.q_est[2]], 
                           [self.q_est[3], 0, -self.q_est[1]], 
                           [-self.q_est[2], self.q_est[1], 0]])
        rotation_matrix = (q_scalar**2 - np.linalg.norm(q_vector, ord=2)**2)*np.identity(3) \
                          - 2 * q_scalar * q_skew + 2 * q_vector @ q_vector.T

        return rotation_matrix
    
    def Quaternion2Euler(self, q):
        euler_angle = np.zeros([3], dtype=np.float)

        euler_angle[0] = math.atan2(2.0*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2)
        euler_angle[1] = -math.asin(np.clip(2.0*(q[1]*q[3]-q[0]*q[2]), -1.0, 1.0))
        euler_angle[2] = math.atan2(2*(q[1]*q[2]+q[0]*q[3]), q[0]**2+q[1]**2-q[2]**2-q[3]**2)

        return euler_angle

    def Euler2Quaternion(self, euler_angle):
        q = np.zeros([4], dtype=np.float)
        
        q[0] = math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2) + \
                math.sin(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
        q[1] = math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2) - \
                math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
        q[2] = math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2) + \
                math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
        q[3] = math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2) - \
                math.sin(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2)

        return q


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    usecols = ['ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)', 'AngleX(deg)',
           'AngleY(deg)', 'AngleZ(deg)']
    data_base = pd.read_csv('../data/201217095118.txt', usecols=usecols ,sep=r'\s+')
    acc_base = data_base.loc[:, ['ax(g)', 'ay(g)', 'az(g)']].values
    omega_base = data_base.loc[:, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']].values
    angle_base = data_base.loc[:, ['AngleX(deg)', 'AngleY(deg)', 'AngleZ(deg)']].values

    MEKF = MultiplicativeEKF(dimension_of_state=3, step_time=0.01)
    angle_filter = []
    for i in range(omega_base.shape[0]):
        angle_filter.append(MEKF.Quaternion2Euler(MEKF.step(omega_base[i, :], acc_base[i, :])))

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