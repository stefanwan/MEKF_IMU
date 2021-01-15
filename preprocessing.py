import re
import numpy as np

with open('../data/original_data.txt', 'r') as fr:
    line = fr.readline()
    data_group = re.split('55 51|55 52|55 53', line)

acc_orig = np.zeros([int((len(data_group)-1)/3), 3], dtype=np.float32)
omega_orig = np.zeros([int((len(data_group)-1)/3), 3], dtype=np.float32)
angle_orig = np.zeros([int((len(data_group)-1)/3), 3], dtype=np.float32)
acc_orig = []
omega_orig = []
angle_orig = []
rounds = 0

for index in range(1, len(data_group)):
    data = data_group[index].split(' ')[1:-4]
    if index == 1 + 3*rounds:
        acc_x = np.int16(int(data[0], 16) + np.int16(int(data[1], 16) * 256)) / 32768 * 16
        acc_y = np.int16(int(data[2], 16) + np.int16(int(data[3], 16) * 256)) / 32768 * 16
        acc_z = np.int16(int(data[4], 16) + np.int16(int(data[5], 16) * 256)) / 32768 * 16
        acc_orig.append([acc_x, acc_y, acc_z])
    elif index == 2 + 3*rounds:
        omega_x = np.int16(int(data[0], 16) + np.int16(int(data[1], 16) * 256)) / 32768 * 2000
        omega_y = np.int16(int(data[2], 16) + np.int16(int(data[3], 16) * 256)) / 32768 * 2000
        omega_z = np.int16(int(data[4], 16) + np.int16(int(data[5], 16) * 256)) / 32768 * 2000
        omega_orig.append([omega_x, omega_y, omega_z])
    elif index == 3 + 3*rounds:
        angle_x = np.int16(int(data[0], 16) + np.int16(int(data[1], 16) * 256)) / 32768 * 180
        angle_y = np.int16(int(data[2], 16) + np.int16(int(data[3], 16) * 256)) / 32768 * 180
        angle_z = np.int16(int(data[4], 16) + np.int16(int(data[5], 16) * 256)) / 32768 * 180
        angle_orig.append([angle_x, angle_y, angle_z])
        rounds += 1

acc_orig = np.asarray(acc_orig, dtype=np.float32)
omega_orig = np.asarray(omega_orig, dtype=np.float32)
angle_orig = np.asarray(angle_orig, dtype=np.float32)