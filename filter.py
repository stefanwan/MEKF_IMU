import numpy as np
import math
import matplotlib.pyplot as plt

dt = 0.01
omegay_ob = []
angley_real = []
accx_ob = []
accz_ob = []
f = open('200528170308y.txt', mode='r')

line = f.readline()
while line:
    str2list = line.split()
    # lens = len(str2list)
    omega_y = str2list[6]
    angle_y = str2list[9]
    acc_x = str2list[2]
    acc_z = str2list[4]
    omegay_ob.append(float(omega_y))
    angley_real.append(float(angle_y))
    accx_ob.append(float(acc_x))
    accz_ob.append(float(acc_z))
    line = f.readline()
f.close()

# system transfer function: x_ = phi*x + g*u + gamma*w
# x = [angle_y, Q_bias], z = [angley_ob]
# phi = [[1, -dt], [0, 1]], g = [[dt], [0]]
# gamma = [[-(dt^2)/2], [dt]]

phi = np.array([[1, -dt],
               [0, 1]], dtype=float)
g = np.array([[dt], [0]], dtype=float)
gamma = np.array([[-(dt**2)/2], [dt]], dtype=float)
h = np.array([1, 0], dtype=int)
p_kk = np.identity(2)
q = 0.1
r = 1
x_filter = []
angley_filter = []
angley_cal = []
q_bias = []
for i in range(len(omegay_ob[:8000])):
    if i == 0:
        x = np.array([[0], [0]])
        u = 0
    else:
        x = x_filter[i-1] 
        u = omegay_ob[i-1]
    acc_x = accx_ob[i]
    acc_z = accz_ob[i] + 1e-5
    z = math.atan(acc_x/(acc_z+1e-5))*180/math.pi
    if acc_x <= 0 and acc_z >= 0:
        z = -z
        # u = u
    elif acc_x <= 0 and acc_z <= 0:
        z = z
        # u = -u  
    elif acc_x >= 0 and acc_z <= 0:
        z = z
        # u = -u
    else:
        z = -z
        # u = u
    angley_cal.append(z)
    p = phi@p_kk@phi.T + gamma*q@gamma.T
    k = p@h.reshape(2, 1)/(h.reshape(1, 2)@p@h.reshape(2, 1)+r)
    x_ = phi@x + g*u
    z_ = h.reshape(1, 2)@x_
    x = x_ + k@(z - z_)
    x_filter.append(x)
    angley_filter.append(x[0][0])
    q_bias.append(x[1][0])
    p_kk = p - (p@h.reshape(2, 1)/(h.reshape(1, 2)@p@h.reshape(2, 1)+r))@h.reshape(1, 2)@p
    
time_series = [t for t in np.arange(0, dt*len(omegay_ob[:8000]), dt)]
plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.plot(time_series, omegay_ob[:8000])
plt.plot(time_series, angley_real[:8000])
plt.plot(time_series, angley_filter)
plt.plot(time_series, angley_cal)
plt.xlabel('time')
plt.ylabel('angle of y')
plt.legend(['real', 'filter', 'cal'])

# plt.subplot(2, 2, 2)
# plt.plot(time_series, angley_real[:8000])
# plt.plot(time_series, angley_filter)
# plt.xlabel('time')
# plt.ylabel('angle of y')

# plt.subplot(2, 2, 3)
# plt.plot(time_series, accx_ob[:8000])
# plt.xlabel('time')
# plt.ylabel('acceleration_x')

# plt.subplot(2, 2, 4)
# plt.plot(time_series, accz_ob[:8000])
# plt.xlabel('time')
# plt.ylabel('acceleration_z')
plt.show()