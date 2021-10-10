import numpy as np
from scipy.optimize import fsolve, leastsq
import matplotlib.pyplot as plt
import pandas as pd

# Problem 1 #########################################################################################################

I_s = 1e-9
N = 1.7
R = 11000
T = 350
V = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.7, 1.8, \
     1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
Q = 1.6021766208e-19
K = 1.380648e-23
guess_1 = 0.1
diode_curr_p1 = []
diode_volt_p1 = []


####################################
# function computes diode current  #
# input v_d diode voltage          #
# output i_diode diode current     #
####################################
def diode_current(v_d, n, temp, i_s):
    i_diode = i_s * (np.exp(v_d * Q / (n * K * temp)) - 1)
    return i_diode


#################################################
# function optimizes diode voltage drop         #
# input v_d (guess) and v source voltage        #
# output err, goes to zero when fsolve succeeds #
#################################################
def diode_voltage(v_d, v, r, n, temp, i_s):
    i_d = diode_current(v_d, n, temp, i_s)
    err = (v_d / r) - (v / r) + i_d
    return err


# Problem 1 Loop
for index in range(len(V)):
    diode_volt = fsolve(diode_voltage, [guess_1], args=(V[index], R, N, T, I_s), xtol=1e-12)  # optimized v_d
    diode_curr = diode_current(diode_volt, N, T, I_s)  # diode current using optimized v_d
    diode_curr_p1.append(diode_curr)
    diode_volt_p1.append(diode_volt)
    # print(diode_volt)  # debug statement
    # print(diode_curr)  # debug statement
    guess_1 = diode_volt  # uses optimized v_d in current part epoch for the next epoch
# print(diode_volt_p1)  # debug statement
# print(diode_curr_p1)  # debug statement

plt.plot(V, diode_curr_p1, label='Source Voltage vs Diode Current')
plt.plot(diode_volt_p1, diode_curr_p1, label='Diode Voltage vs Diode Current')
plt.ylabel('Diode Current (log scale)')
plt.xlabel('Voltage')
plt.yscale('log')
plt.legend()
plt.show()

# Problem 2 ##########################################################################################################
diodeIV = pd.read_csv('DiodeIV.txt', sep=' ', names=['Voltage', 'Current'])  # read in measured data and assign cols
# print(diodeIV)  # debug statement
diode_volt_2 = diodeIV['Voltage'].copy()
# print(diode_volt_2)  # debug statement
source_v = diode_volt_2.to_numpy()  # created array of measured voltages
diode_curr_2 = diodeIV['Current'].copy()
meas_diode_i = diode_curr_2.to_numpy()  # create array of measured currents

P2_AREA = 1e-8  # cross sectional area of diode
r_val = np.array([10000])  # initial guess of resistor value in Ohms
p_val = 0.8  # initial guess of Phi
n_val = 1.5  # initial guess of ideality factor (n)
P2_T = 375  # temp of diode


################################################################################
# This function does the optimization for the resistor                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_r(r_value, ide_value, phi_value, area, temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)  # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)  # an array to hold the diode currents
    prev_v = 0.1  # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / (K * temp))

    for index in range(len(src_v)):
        prev_v = fsolve(diode_voltage, prev_v, (src_v[index], r_value, ide_value, temp, is_value), xtol=1e-12)[0]
        est_v[index] = prev_v  # store for error analysis

    # compute the diode current
    diode_i = diode_current(est_v, ide_value, temp, is_value)
    return meas_i - diode_i


################################################################################
# This function does the optimization for the ideality factor                  #
# Inputs:                                                                      #
#    r_val     - optimised value of the resistor                               #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_n(ide_value, r_value, phi_value, area, temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)  # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)  # an array to hold the diode currents
    prev_v = 0.1  # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / (K * temp))

    for index in range(len(src_v)):
        prev_v = fsolve(diode_voltage, prev_v, (src_v[index], r_value, ide_value, temp, is_value), xtol=1e-12)[0]
        est_v[index] = prev_v  # store for error analysis

    # compute the diode current
    diode_i = diode_current(est_v, ide_value, temp, is_value)
    return (meas_i - diode_i) / (meas_i + diode_i + 1e-15)


################################################################################
# This function does the optimization for Phi                                  #
# Inputs:                                                                      #
#    r_val     - optimised value of the resistor                               #
#    n_val     - optimised value of the ideality                               #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_p(phi_value, r_value, n_value, area, temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)  # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)  # an array to hold the diode currents
    prev_v = 0.1  # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / (K * temp))

    for index in range(len(src_v)):
        prev_v = fsolve(diode_voltage, prev_v, (src_v[index], r_value, n_value, temp, is_value), xtol=1e-12)[0]
        est_v[index] = prev_v  # store for error analysis

    # compute the diode current
    diode_i = diode_current(est_v, n_value, temp, is_value)
    return (meas_i - diode_i) / (meas_i + diode_i + 1e-15)


# this block loops through to improve the values
err = 1
iteration = 0
while err > 1e-13:
    r_val_opt = leastsq(opt_r, r_val, args=(n_val, p_val, P2_AREA, P2_T, source_v, meas_diode_i))
    r_val = r_val_opt[0][0]

    n_val_opt = leastsq(opt_n, n_val, args=(r_val, p_val, P2_AREA, P2_T, source_v, meas_diode_i))
    n_val = n_val_opt[0][0]

    p_val_opt = leastsq(opt_p, p_val, args=(r_val, n_val, P2_AREA, P2_T, source_v, meas_diode_i))
    p_val = p_val_opt[0][0]

    res = opt_p(p_val, r_val, n_val, P2_AREA, P2_T, source_v, meas_diode_i)
    err = np.sum(np.abs(res)) / len(res)
    print('Iteration:', iteration + 1)
    print('Resistor value', r_val)
    print('Ideality factor', n_val)
    print('Phi', p_val)
    print('Error:', err)

    iteration += 1

# this block of code uses the optimized values found in the least squares to plot a new diode VI curve
is_value = P2_AREA * P2_T * P2_T * np.exp(-p_val * Q / (K * P2_T))  # saturation current of diode
diode_volt_P2_ = []  # list to hold V for plotting
diode_curr_P2_ = []  # list to hold i for plotting
guess_1_ = 0.1  # starting guess v_d
for index in range(len(source_v)):
    diode_volt_P2 = fsolve(diode_voltage, [guess_1_], args=(source_v[index], r_val, n_val, P2_T, is_value), xtol=1e-12)
    diode_curr_P2 = diode_current(diode_volt_P2, n_val, P2_T, is_value)  # diode current using optimized v_d
    diode_curr_P2_.append(diode_curr_P2)
    diode_volt_P2_.append(diode_volt_P2)
    guess_1_ = diode_volt_P2  # uses optimized v_d in current part epoch for the next epoch

# plots the two curves
plt.plot(source_v, meas_diode_i, linestyle='solid', marker='o', label='Source Voltage vs Measured Diode Current')
plt.plot(source_v, diode_curr_P2_, linestyle='solid', marker='+', label='Source Voltage vs Model Diode Current')
plt.ylabel('Diode Current (log scale)')
plt.xlabel('Source Voltage')
plt.yscale('log')
plt.legend()
plt.show()

