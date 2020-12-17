import numpy as np
from math import sin, cos, atan2, sqrt, asin
from scipy.linalg import expm

########## RotateMatrix integration ##########

def Euler2RotateMatrix(euler_angle):
    rotate_matrix = np.zeros([3, 3])
    euler_angle = euler_angle / 180 * np.pi
    # rotate_matrix[0, 0] = cos(euler_angle[1]) * cos(euler_angle[2])
    # rotate_matrix[0, 1] = -cos(euler_angle[1]) * sin(euler_angle[2])
    # rotate_matrix[0, 2] = sin(euler_angle[1])
    # rotate_matrix[1, 0] = sin(euler_angle[0]) * sin(euler_angle[1]) * cos(euler_angle[2]) + \
    #     cos(euler_angle[0]) * sin(euler_angle[2])
    # rotate_matrix[1, 1] = -sin(euler_angle[0]) * sin(euler_angle[1]) * sin(euler_angle[2]) + \
    #     cos(euler_angle[0]) * cos(euler_angle[2])
    # rotate_matrix[1, 2] = -sin(euler_angle[0]) * cos(euler_angle[1])
    # rotate_matrix[2, 0] = -cos(euler_angle[0]) * sin(euler_angle[1]) * cos(euler_angle[2]) + \
    #     sin(euler_angle[0]) * sin(euler_angle[2])
    # rotate_matrix[2, 1] = cos(euler_angle[0]) * sin(euler_angle[1]) * sin(euler_angle[2]) + \
    #     sin(euler_angle[0]) * cos(euler_angle[2])
    # rotate_matrix[2, 2] = cos(euler_angle[0]) * cos(euler_angle[1])

    rotate_matrix[0, 0] = cos(euler_angle[1]) * cos(euler_angle[2])
    rotate_matrix[0, 1] = -cos(euler_angle[0]) * sin(euler_angle[2]) + \
        sin(euler_angle[0]) * sin(euler_angle[1]) * cos(euler_angle[2])
    rotate_matrix[0, 2] = sin(euler_angle[0]) * sin(euler_angle[2]) + \
        cos(euler_angle[2]) * sin(euler_angle[1]) * cos(euler_angle[0])
    rotate_matrix[1, 0] = cos(euler_angle[1]) * sin(euler_angle[2])
    rotate_matrix[1, 1] = cos(euler_angle[0]) * cos(euler_angle[2]) + \
        sin(euler_angle[0]) * sin(euler_angle[1]) * sin(euler_angle[2])
    rotate_matrix[1, 2] = -sin(euler_angle[0]) * cos(euler_angle[2])	+ \
        sin(euler_angle[1]) * sin(euler_angle[2]) * cos(euler_angle[0])
    rotate_matrix[2, 0] = -sin(euler_angle[1])
    rotate_matrix[2, 1] = sin(euler_angle[0]) * cos(euler_angle[1])
    rotate_matrix[2, 2] = cos(euler_angle[0]) * cos(euler_angle[1])

    return rotate_matrix

def RotateMatrix2Euler(rotate_matrix):
    euler_angle = np.zeros([3])
    # euler_angle[0] = -atan2(rotate_matrix[1, 2], rotate_matrix[2, 2])
    # euler_angle[2] = -atan2(rotate_matrix[0, 1], rotate_matrix[0, 0])
    # euler_angle[1] = atan2(rotate_matrix[0, 2], (cos(euler_angle[0])*rotate_matrix[2, 2] - \
    #                     sin(euler_angle[0])*rotate_matrix[1, 2]))

    euler_angle[0] = atan2(rotate_matrix[2, 1], rotate_matrix[2, 2])
    euler_angle[2] = atan2(rotate_matrix[1, 0], rotate_matrix[0, 0])
    euler_angle[1] = -atan2(rotate_matrix[2, 0],sqrt(rotate_matrix[1, 0]*rotate_matrix[1, 0] + \
                        rotate_matrix[0, 0]*rotate_matrix[0, 0]))
    # euler_angle = euler_angle / np.pi * 180

    return euler_angle

omega_vector = np.array([[0.0],
                         [0.0],
                         [-20/180*np.pi]])
omega = np.array([[0, 20/180*np.pi, 0],
                  [-20/180*np.pi, 0, 0],
                  [0, 0, 0]], dtype=np.float32)
angle = np.array([.0, 90, .0], dtype=np.float32)
R = Euler2RotateMatrix(angle)
temp1 = expm(omega)

#### 指数映射的简便计算，精度更高####
theta = np.linalg.norm(omega_vector, ord=2)
omega_vector = omega_vector/theta
omega_times = np.array([[0, -omega_vector[2, 0], omega_vector[1, 0]],
                        [omega_vector[2, 0], 0, -omega_vector[0, 0]],
                        [-omega_vector[1, 0], omega_vector[0, 0], 0]])
temp2 = np.identity(3)*cos(theta) + sin(theta)*omega_times + (1 - cos(theta))*(omega_vector@omega_vector.T)

R1 = R @ temp1
R2 = R @ temp2
angle = RotateMatrix2Euler(R1)
print(angle/np.pi*180)
angle = RotateMatrix2Euler(R2)
print(angle/np.pi*180)

########## Quaternion integration ##########

def Euler2Quaternion(euler_angle):
    euler_angle = euler_angle / 180 * np.pi
    q = np.zeros([4], dtype=np.float64)
    
    q[0] = cos(euler_angle[0]/2)*cos(euler_angle[1]/2)*cos(euler_angle[2]/2) + \
            sin(euler_angle[0]/2)*sin(euler_angle[1]/2)*sin(euler_angle[2]/2)
    q[1] = sin(euler_angle[0]/2)*cos(euler_angle[1]/2)*cos(euler_angle[2]/2) - \
            cos(euler_angle[0]/2)*sin(euler_angle[1]/2)*sin(euler_angle[2]/2)
    q[2] = cos(euler_angle[0]/2)*sin(euler_angle[1]/2)*cos(euler_angle[2]/2) + \
            sin(euler_angle[0]/2)*cos(euler_angle[1]/2)*sin(euler_angle[2]/2)
    q[3] = cos(euler_angle[0]/2)*cos(euler_angle[1]/2)*sin(euler_angle[2]/2) - \
            sin(euler_angle[0]/2)*sin(euler_angle[1]/2)*cos(euler_angle[2]/2)

    return q

def Quaternion2Euler(q):
    euler_angle = np.zeros([3], dtype=np.float64)

    euler_angle[0] = atan2(2.0*(q[2]*q[3]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2)
    euler_angle[1] = -asin(np.clip(2.0*(q[1]*q[3]-q[0]*q[2]), -1.0, 1.0))
    euler_angle[2] = atan2(2*(q[1]*q[2]+q[0]*q[3]), q[0]**2+q[1]**2-q[2]**2-q[3]**2)

    return euler_angle

def ExpQua(v):
    q = np.zeros([4], dtype=np.float64)

    phi = np.linalg.norm(v, ord=2)
    q[0] = cos(phi)
    q[1:] = v/phi*sin(phi) 

    return q

def MulQua(q1, q2):
    q = np.zeros([4], dtype=np.float64)

    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return q

# angle = np.array([0, 90.0, 0.0])
# gyro = np.array([0.0, 0, -20/180*np.pi])
# q = Euler2Quaternion(angle)
# q = MulQua(q, ExpQua(gyro*.5))
# angle = Quaternion2Euler(q)
# print(angle/np.pi*180)
