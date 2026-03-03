# Nonlinear_planner_robot_ex_filled.py
#
# Filled version of the exercise template Nonlinear_planner_robot_ex.py (Δv formulation).
# IMPORTANT for MPCrun.py:
#   - dt and H MUST match dt_mpc and H inside MPCrun.py when you export nlp_circle_wall_obstacle_dv.casadi.
#
# Exports: nlp_circle_wall_obstacle_dv.casadi
#
# Note: MPCrun.py (as provided) loads nlp_circle_wall_obstacle_dv_slack.casadi by default.
# If you want to use this file directly with MPCrun.py, either:
#   (a) change MPCrun.py's nlp_solver_file to 'nlp_circle_wall_obstacle_dv.casadi', OR
#   (b) rename the output .casadi accordingly.

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def main():
    dt = 0.1  # Sampling time
    H = 20    # horizon (set to 20 to match MPCrun.py defaults)

    use_rk4 = True  # true: RK4, false: Euler

    nx = 6  # x = [x1; x2; psi; v1_prev; v2_prev; omega_prev]
    nu = 3  # u = [Δv1; Δv2; Δomega]
    nz = 2  # z = [x1; x2] (track position)

    b_wall = 0.0  # Wall constraint

    p_obs = np.array([4.0, b_wall], dtype=float)  # Obstacle
    r_obs = 0.55  # Obstacle radius
    r_safe = r_obs  # Safety radius

    # Bounds (match MPCrun.py defaults)
    dv_max = 2.0 * dt
    domega_max = 3.0 * dt
    u_max = np.array([dv_max, dv_max, domega_max], dtype=float)

    v_max = 3.0
    omega_max = 3.0

    # Weights (tune in the exercise)
    Q = np.diag([10.0, 10.0])              # tracking
    R = np.diag([0.2, 0.2, 0.05])          # penalize Δv
    Q_T = 5.0 * Q                          # terminal

    # Start outside the circle (and feasible wrt wall)
    radius = 4.0
    center = np.array([0.0, 0.0], dtype=float)
    x0_val = np.array([radius + 2.0, b_wall + 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Reference (circle) sampled at k=1,...,H
    w = 2 * 2 * np.pi / (200 * dt)
    Gamma_val = np.zeros((nz * H, 1))
    for k in range(1, H + 1):
        t = k * dt
        rk = center + radius * np.array([np.cos(w * t), np.sin(w * t)])
        Gamma_val[(k - 1) * nz : k * nz, 0] = rk

    # 1) Continuous-time kinematics for pose xp=[x1;x2;psi], input v=[v1;v2;omega]
    xp = ca.MX.sym("xp", 3, 1)
    v = ca.MX.sym("v", 3, 1)

    psi = xp[2]
    c = ca.cos(psi)
    s = ca.sin(psi)

    x1_dot = c * v[0] - s * v[1]
    x2_dot = s * v[0] + c * v[1]
    psi_dot = v[2]

    xpdot = ca.vertcat(x1_dot, x2_dot, psi_dot)
    f_ct = ca.Function("f_ct", [xp, v], [xpdot])

    # 2) Discrete-time integrator for pose: xp(k+1)=F_dt(xp(k), v(k))
    dt_sym = ca.MX.sym("dt", 1, 1)

    F_euler = ca.Function("F_euler", [xp, v, dt_sym], [xp + dt_sym * f_ct(xp, v)])

    k1 = f_ct(xp, v)
    k2 = f_ct(xp + (dt_sym / 2) * k1, v)
    k3 = f_ct(xp + (dt_sym / 2) * k2, v)
    k4 = f_ct(xp + dt_sym * k3, v)
    F_rk4 = ca.Function("F_rk4", [xp, v, dt_sym], [xp + (dt_sym / 6) * (k1 + 2 * k2 + 2 * k3 + k4)])

    F_dt = F_rk4 if use_rk4 else F_euler

    # 3) Multiple-shooting NLP in CasADi Opti
    opti = ca.Opti("nlp")

    # Parameters
    x0 = opti.parameter(nx, 1)
    Gamma = opti.parameter(nz * H, 1)
    bw = opti.parameter(1, 1)

    # Decision variables
    X = opti.variable(nx, H + 1)
    U = opti.variable(nu, H)

    # Initial conditions
    opti.subject_to(X[:, 0] == x0)

    # Dynamics + Objective J
    J = 0
    for k in range(H):
        # Velocity state update (Δv)
        v_prev = X[3:6, k]
        dvk = U[:, k]
        v_k = v_prev + dvk

        # Pose update with commanded velocity held constant over dt
        xp_next = F_dt(X[0:3, k], v_k, dt)

        # Dynamics constraints
        opti.subject_to(X[0:3, k + 1] == xp_next)
        opti.subject_to(X[3:6, k + 1] == v_k)

        # Tracking cost (use position at k+1)
        rk = Gamma[k * nz : (k + 1) * nz]
        zk = X[0:2, k + 1]
        ek = rk - zk
        J = J + ca.mtimes([ek.T, ca.DM(Q), ek]) + ca.mtimes([dvk.T, ca.DM(R), dvk])

    # Terminal cost at time H (track position)
    rH = Gamma[(H - 1) * nz : H * nz]
    zH = X[0:2, H]
    eH = rH - zH
    J = J + ca.mtimes([eH.T, ca.DM(Q_T), eH])

    # Wall constraint for k=1..H (nodes 1..H in X)
    opti.subject_to(X[1, 1:] >= bw)

    # Obstacle avoidance for k=1..H
    dist2 = (X[0, 1:] - p_obs[0]) ** 2 + (X[1, 1:] - p_obs[1]) ** 2
    opti.subject_to(dist2 >= r_safe**2)

    # Bounds on commanded velocities for k=0..H-1 (stored in X[3:6,1:])
    opti.subject_to(X[3, 1:] <= v_max)
    opti.subject_to(X[3, 1:] >= -v_max)
    opti.subject_to(X[4, 1:] <= v_max)
    opti.subject_to(X[4, 1:] >= -v_max)
    opti.subject_to(X[5, 1:] <= omega_max)
    opti.subject_to(X[5, 1:] >= -omega_max)

    # Bounds on Δv inputs
    opti.subject_to(U[0, :] <= u_max[0])
    opti.subject_to(U[0, :] >= -u_max[0])
    opti.subject_to(U[1, :] <= u_max[1])
    opti.subject_to(U[1, :] >= -u_max[1])
    opti.subject_to(U[2, :] <= u_max[2])
    opti.subject_to(U[2, :] >= -u_max[2])

    opti.minimize(J)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    solve_nlp = opti.to_function(
        "solve_nlp",
        [x0, Gamma, bw, X, U],
        [U, X, opti.f],
        ["x0", "Gamma", "b_wall", "Xguess", "Uguess"],
        ["U_opt", "X_opt", "J_opt"],
    )
    solve_nlp.save("nlp_circle_wall_obstacle_dv.casadi")
    print("Saved: nlp_circle_wall_obstacle_dv.casadi")

    # Solve once (demo)
    solve_nlp = ca.Function.load("nlp_circle_wall_obstacle_dv.casadi")

    Xguess = np.zeros((nx, H + 1))
    Uguess = np.zeros((nu, H))

    Xguess[:, 0] = x0_val
    ref = Gamma_val.reshape(H, nz).T
    Xguess[0, 1:] = ref[0, :]
    Xguess[1, 1:] = ref[1, :]

    Xguess[1, :] = np.maximum(Xguess[1, :], b_wall + 0.15)

    # Add a small "bump" around the obstacle to help initial feasibility
    k_mid = int(round(H / 2))
    bump = np.exp(-((np.arange(H + 1) - k_mid) ** 2) / (2 * (0.12 * H) ** 2))
    Xguess[1, :] = Xguess[1, :] + (r_safe + 0.25) * bump

    U_sol, X_sol, J_sol = solve_nlp(x0_val.reshape(-1, 1), Gamma_val, np.array([[b_wall]]), Xguess, Uguess)

    X_opt = np.array(X_sol.full())
    print(f"NLP solved. J = {float(J_sol):.6e}")

    x1_traj = X_opt[0, :]
    x2_traj = X_opt[1, :]
    psi_traj = X_opt[2, :]

    plt.figure(figsize=(7, 7))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    tt = np.linspace(0, H * dt, 400)
    circ = center.reshape(2, 1) + radius * np.vstack((np.cos(w * tt), np.sin(w * tt)))
    plt.plot(circ[0, :], circ[1, :], "--", linewidth=1.2)

    xx = np.array([min(np.min(x1_traj), np.min(circ[0, :])) - 2, max(np.max(x1_traj), np.max(circ[0, :])) + 2])
    plt.plot(xx, b_wall * np.ones_like(xx), "k-", linewidth=2)

    th = np.linspace(0, 2 * np.pi, 240)
    plt.plot(p_obs[0] + r_safe * np.cos(th), p_obs[1] + r_safe * np.sin(th), "k-", linewidth=2)

    plt.plot(x1_traj, x2_traj, "-", linewidth=2)
    plt.show()


if __name__ == "__main__":
    main()
