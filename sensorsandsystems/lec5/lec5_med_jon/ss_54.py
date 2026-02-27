import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Parameters
# ===============================
Ts = 0.001
phi = 0.9

sigma_wa = 0.1
sigma_wb = 0
sigma_v  = 0.1

bx_true = -0.2
by_true = 0.15

N = 50000
time = np.arange(N) * Ts

# ===============================
# TRUE SYSTEM (2D, 6 states)
# States per axis: [x, v, a]
# ===============================

x_true = np.zeros((6, N))

# Acceleration inputs (2D motion)
w1 = 2*np.pi*0.1
w2 = 2*np.pi*1

ax_input = np.sin(w1*time) + 0.7*np.sin(w2*time)
ay_input = 0.8*np.cos(w1*time) + 0.5*np.sin(0.5*w2*time)

# ===============================
# KALMAN FILTER (8 states)
# States per axis: [x, v, a, b]
# ===============================

Phi1D = np.array([
    [1, Ts, 0.5*Ts**2, 0],
    [0, 1, Ts, 0],
    [0, 0, phi, 0],
    [0, 0, 0, 1]
])

Phi = np.block([
    [Phi1D, np.zeros((4,4))],
    [np.zeros((4,4)), Phi1D]
])

Q1D = np.diag([0, 0, sigma_wa**2, sigma_wb**2])
Q = np.block([
    [Q1D, np.zeros((4,4))],
    [np.zeros((4,4)), Q1D]
])

H = np.array([
    [0,0,1,1, 0,0,0,0],
    [0,0,0,0, 0,0,1,1]
])

R = sigma_v**2 * np.eye(2)

x_hat = np.zeros((8, N))
P = np.eye(8)

# ===============================
# Simulation + KF
# ===============================

for k in range(N-1):

    # ---- TRUE SYSTEM ----
    # X axis
    x_true[0,k+1] = x_true[0,k] + Ts*x_true[1,k] + 0.5*Ts**2*ax_input[k]
    x_true[1,k+1] = x_true[1,k] + Ts*ax_input[k]
    x_true[2,k+1] = ax_input[k]

    # Y axis
    x_true[3,k+1] = x_true[3,k] + Ts*x_true[4,k] + 0.5*Ts**2*ay_input[k]
    x_true[4,k+1] = x_true[4,k] + Ts*ay_input[k]
    x_true[5,k+1] = ay_input[k]

    # measurement (2D accelerometer)
    y = np.array([
        ax_input[k] + bx_true,
        ay_input[k] + by_true
    ]) + sigma_v*np.random.randn(2)

    # ---- KALMAN FILTER ----
    x_pred = Phi @ x_hat[:,k]
    P_pred = Phi @ P @ Phi.T + Q

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    innovation = y - H @ x_pred
    x_hat[:,k+1] = x_pred + K @ innovation
    P = (np.eye(8) - K @ H) @ P_pred

# ===============================
# Plot Results
# ===============================

plt.figure(figsize=(12,10))
plt.suptitle("2D Accelerometer Kalman Filter")

# 2D Position Trajectory
plt.subplot(3,1,1)
plt.plot(x_true[0,:], x_true[3,:], label="True")
plt.plot(x_hat[0,:], x_hat[4,:], '--', label="Estimated")
plt.title("2D Position")
plt.legend()
plt.axis("equal")
plt.grid()

# X bias
plt.subplot(3,1,2)
plt.plot(time, bx_true*np.ones(N), label="True bx")
plt.plot(time, x_hat[3,:], '--', label="Estimated bx")
plt.legend()
plt.grid()

# Y bias
plt.subplot(3,1,3)
plt.plot(time, by_true*np.ones(N), label="True by")
plt.plot(time, x_hat[7,:], '--', label="Estimated by")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()