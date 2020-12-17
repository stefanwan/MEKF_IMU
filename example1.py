import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def Euler2Quaternion(euler_angle):
    q = np.zeros([4], dtype=np.float64)
    
    q[0] = math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2) + \
            math.sin(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
    q[1] = math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.cos(euler_angle[2]/2) - \
            math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
    q[2] = math.cos(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2) + \
            math.sin(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2)
    q[3] = math.cos(euler_angle[0]/2)*math.cos(euler_angle[1]/2)*math.sin(euler_angle[2]/2) - \
            math.sin(euler_angle[0]/2)*math.sin(euler_angle[1]/2)*math.cos(euler_angle[2]/2)

    return q

def Quaternion2Euler(q):
    euler_angle = np.zeros([3], dtype=np.float64)

    euler_angle[0] = math.atan2(2.0*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2)
    euler_angle[1] = -math.asin(np.clip(2.0*(q[1]*q[3]-q[0]*q[2]), -1.0, 1.0))
    euler_angle[2] = math.atan2(2*(q[1]*q[2]+q[0]*q[3]), q[0]**2+q[1]**2-q[2]**2-q[3]**2)

    return euler_angle

def ExpQua(v):
    q = np.zeros([4], dtype=np.float64)

    phi = np.linalg.norm(v, ord=2)
    if phi == 0:
        q = np.array([1, 0, 0, 0])
    else:
        q[0] = math.cos(phi)
        q[1:] = v/phi*math.sin(phi) 

    return q

def MulQua(q1, q2):
    q = np.zeros([4], dtype=np.float64)

    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return q

def InvQua(q):
    q_inv = np.zeros([4], dtype=np.float64)

    q_inv[0] = q[0]
    q_inv[1:4] = -q[1:4]

    return q_inv

def ExpRot(v):
    normv = np.linalg.norm(v, ord=2)

    s = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    if normv == 0:
        RotM = np.identity(3)
    else:
        RotM = np.identity(3) + math.sin(normv)/normv*s + (1-math.cos(normv))/normv**2*(s@s)
    
    return RotM

def WraptoPi(x):
    xwrap = x % (2*np.pi)
    if abs(xwrap) > np.pi:
        xwrap = xwrap - 2*np.pi*np.sign(xwrap)

    return xwrap

def LogQua(q):
    norm_qv = np.linalg.norm(q[1:], ord=2)
    # if abs(q[0]-1) <= 1e-5:
    if q[0] == 1:
        u = np.array([[1],
                      [0],
                      [0]])
        theta = 0
    else:
        u = q[1:]/norm_qv
        theta = WraptoPi(math.atan2(norm_qv, q[0])*2)
    v = u*theta

    return v


usecols = ['ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)', 'AngleX(deg)',
           'AngleY(deg)', 'AngleZ(deg)']
data_base = pd.read_csv('../data/201217111345.txt', usecols=usecols ,sep=r'\s+')
acc_base = data_base.loc[:, ['ax(g)', 'ay(g)', 'az(g)']].values
omega_base = data_base.loc[:, ['wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']].values
angle_base = data_base.loc[:, ['AngleX(deg)', 'AngleY(deg)', 'AngleZ(deg)']].values

dt = 0.01
Q_angle = 0.001
# Q_bias = 0.03
R_measure = 0.1

# Q = np.array([[Q_angle, 0, 0, 0],
#               [0, Q_angle, 0, 0],
#               [0, 0, Q_angle, 0],
#               [0, 0, 0, Q_angle]])
# R = np.array([[R_measure, 0, 0, 0],
#               [0, R_measure, 0, 0],
#               [0, 0, R_measure, 0],
#               [0, 0, 0, R_measure]])
# angle = np.array([[0.0],
#                   [0.0],
#                   [0.0]], dtype=np.float64)
# p_kk = np.zeros([4, 4], dtype=np.float64)
# q_est = Euler2Quaternion(angle)

# angle_filter = []
# for i in range(omega_base.shape[0]-1):
#     # gx = 0.5*(omega_base[i, 0]+omega_base[i+1, 0])/180*np.pi
#     # gy = 0.5*(omega_base[i, 1]+omega_base[i+1, 0])/180*np.pi
#     # gz = 0.5*(omega_base[i, 2]+omega_base[i+1, 0])/180*np.pi
#     gx = omega_base[i, 0]/180*np.pi
#     gy = omega_base[i, 1]/180*np.pi
#     gz = omega_base[i, 2]/180*np.pi
#     u = np.array([gx, gy, gz], dtype = np.float64)
#     dq = ExpQua(u*0.5*dt)
#     q_est = MulQua(q_est, dq)

#     Phi = np.array([[dq[0], -dq[1], -dq[2], -dq[3]],
#                     [dq[1], dq[0], dq[3], -dq[2]],
#                     [dq[2], -dq[3], dq[0], dq[1]],
#                     [dq[3], dq[2], -dq[1], dq[0]]], dtype=np.float64)
#     G = np.array([[q_est[0], -q_est[1], -q_est[2], -q_est[3]],
#                   [q_est[1], q_est[0], -q_est[3], q_est[2]],
#                   [q_est[2], q_est[3], q_est[0], -q_est[1]],
#                   [q_est[3], -q_est[2], q_est[1], q_est[0]]], dtype=np.float64)
    
#     p = Phi @ p_kk @ Phi.T + G @ Q * dt  @ G.T
#     K = p @ np.linalg.inv(p + R)
#     angle_est = Quaternion2Euler(q_est)
#     z = np.array([[math.atan2(acc_base[i, 1], acc_base[i, 2])],
#                   [math.atan(-acc_base[i, 0]/math.sqrt(acc_base[i, 1]**2+acc_base[i, 2]**2))],
#                   [angle_est[2]]], dtype=np.float64)
#     qmea = Euler2Quaternion(z)
#     dq = qmea - q_est
#     q_est = q_est + K @ dq
#     q_est = q_est / np.linalg.norm(q_est, ord=2)
#     p_kk = (np.identity(4) - K) @ p
#     angle_filter.append(Quaternion2Euler(q_est))

Q = np.array([[Q_angle, 0, 0],
              [0, Q_angle, 0],
              [0, 0, Q_angle]])
R = np.array([[R_measure, 0, 0],
              [0, R_measure, 0],
              [0, 0, R_measure]])
angle = np.array([[0.0],
                  [0.0],
                  [0.0]], dtype=np.float64)
p_kk = np.zeros([3, 3], dtype=np.float64)
q_est = Euler2Quaternion(angle)

angle_filter = []
for i in range(omega_base.shape[0]):
    # gx = 0.5*(omega_base[i, 0]+omega_base[i+1, 0])/180*np.pi
    # gy = 0.5*(omega_base[i, 1]+omega_base[i+1, 0])/180*np.pi
    # gz = 0.5*(omega_base[i, 2]+omega_base[i+1, 0])/180*np.pi
    gx = omega_base[i, 0]/180*np.pi
    gy = omega_base[i, 1]/180*np.pi
    gz = omega_base[i, 2]/180*np.pi
    u = np.array([gx, gy, gz], dtype = np.float64)
    dq = ExpQua(u*0.5*dt)
    q_est = MulQua(q_est, dq)

    Phi = ExpRot(u*dt)
    
    p = Phi @ p_kk @ Phi.T + Q * dt
    K = p @ np.linalg.inv(p + R)
    angle_est = Quaternion2Euler(q_est)
    z = np.array([[math.atan2(acc_base[i, 1], acc_base[i, 2])],
                  [math.atan(-acc_base[i, 0]/math.sqrt(acc_base[i, 1]**2+acc_base[i, 2]**2))],
                  [angle_est[2]]], dtype=np.float64)
    qmea = Euler2Quaternion(z)
    dv = K @ LogQua(MulQua(InvQua(q_est), qmea))
    q_est = MulQua(q_est, ExpQua(dv))
    p_kk = (np.identity(3) - K) @ p
    angle_filter.append(Quaternion2Euler(q_est))

angle_filter = np.asarray(angle_filter)
plt.figure()
plt.plot(angle_filter[:, 0]/np.pi*180)
plt.figure()
plt.plot(angle_base[:, 0])
# plt.legend(['filter', 'base'])
plt.figure()
plt.plot(angle_filter[:, 1]/np.pi*180)
# plt.figure()
plt.plot(angle_base[:, 1])
plt.legend(['filter', 'base'])

plt.figure()
plt.plot(angle_filter[:, 2]/np.pi*180)
plt.figure()
plt.plot(angle_base[:, 2])
plt.show()