import numpy as np
from scipy.signal import lti, lsim
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------
# Robot simulation (unchanged)
# -------------------------------------------------
A = np.array([[0.0, 1.0],
              [0.0, 0.0]])
B = np.array([[0.0],
              [1.0]])
C = np.array([[1.0, 0.0]])
D = 0.0

system = lti(A, B, C, D)

dt = 0.01
T = 28
t = np.arange(0, T, dt)

v_des = 0.8
omega_circle = 0.4
omega_quarter = (np.pi/2) / 4.0

kv = 3.0
kw = 6.0

u_v = np.zeros_like(t)
u_w = np.zeros_like(t)

v_sim = 0
w_sim = 0

for i, ti in enumerate(t):
    if ti < 4:
        v_ref = v_des; w_ref = 0
    elif ti < 16:
        v_ref = v_des; w_ref = omega_circle
    elif ti < 20:
        v_ref = v_des; w_ref = 0
    elif ti < 24:
        v_ref = v_des; w_ref = omega_quarter
    else:
        v_ref = v_des; w_ref = 0

    a_v = kv * (v_ref - v_sim)
    a_w = kw * (w_ref - w_sim)

    u_v[i] = a_v
    u_w[i] = a_w

    v_sim += a_v * dt
    w_sim += a_w * dt

_, _, xv = lsim(system, U=u_v, T=t)
v = xv[:, 1]

_, _, xw = lsim(system, U=u_w, T=t)
omega = xw[:, 1]

v_fun = interp1d(t, v, fill_value="extrapolate")
omega_fun = interp1d(t, omega, fill_value="extrapolate")

def kinematics(state, time):
    x, y, phi = state
    dx = v_fun(time) * np.cos(phi)
    dy = v_fun(time) * np.sin(phi)
    dphi = omega_fun(time)
    return [dx, dy, dphi]

pose = odeint(kinematics, [0,0,0], t)
x_true = pose[:,0]
y_true = pose[:,1]
phi = pose[:,2]

# -------------------------------------------------
# TRUE BODY-FRAME ACCELERATION
# -------------------------------------------------
a_forward = u_v          # longitudinal acceleration
a_lateral = np.zeros_like(t)  # lateral acceleration (assume zero)

# Add measurement noise
sigma_v = 0.01
a_forward_meas = a_forward + sigma_v*np.random.randn(len(t))
a_lateral_meas = a_lateral + sigma_v*np.random.randn(len(t))

# -------------------------------------------------
# 2D Kalman Filter (body-frame integration)
# States: x, y, v_fwd, v_lat, bx, by
# -------------------------------------------------
N = len(t)
x_hat = np.zeros((6,N))
P = np.eye(6)
Q = np.diag([0,0,1e-6,1e-6,1e-6,1e-6])  # small process noise
R = sigma_v**2 * np.eye(2)               # measurement noise

for k in range(N-1):
    # ---- Body-frame accelerations with bias correction ----
    ax = a_forward_meas[k] - x_hat[4,k]  # subtract bias
    ay = a_lateral_meas[k] - x_hat[5,k]

    # ---- Update velocities ----
    x_hat[2,k+1] = x_hat[2,k] + ax*dt
    x_hat[3,k+1] = x_hat[3,k] + ay*dt

    # ---- Rotate velocity to global frame for position ----
    cos_phi = np.cos(phi[k])
    sin_phi = np.sin(phi[k])
    vx_global = cos_phi*x_hat[2,k+1] - sin_phi*x_hat[3,k+1]
    vy_global = sin_phi*x_hat[2,k+1] + cos_phi*x_hat[3,k+1]

    # ---- Update positions ----
    x_hat[0,k+1] = x_hat[0,k] + vx_global*dt
    x_hat[1,k+1] = x_hat[1,k] + vy_global*dt

    # ---- Bias estimation (simple KF step) ----
    # Innovation: measured acceleration minus predicted (with bias)
    innovation = np.array([a_forward_meas[k]-x_hat[2,k+1],
                           a_lateral_meas[k]-x_hat[3,k+1]])
    K = np.zeros((6,2))
    K[4,0] = 0.001
    K[5,1] = 0.001
    x_hat[:,k+1] += K @ innovation

# -------------------------------------------------
# Plot results
# -------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(x_true, y_true, label="True robot path")
plt.plot(x_hat[0,:], x_hat[1,:], '--', label="KF estimate")
plt.axis("equal")
plt.legend()
plt.grid()
plt.title("Robot Motion vs 2D Accelerometer KF (Body-frame Integration)")
plt.show()

# Plot estimated bias
plt.figure(figsize=(8,4))
plt.plot(t, x_hat[4,:], label="Estimated bx")
plt.plot(t, x_hat[5,:], label="Estimated by")
plt.title("Estimated accelerometer bias")
plt.legend()
plt.grid()
plt.show()