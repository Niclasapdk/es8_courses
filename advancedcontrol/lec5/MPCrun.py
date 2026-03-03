# MPC simulation using the Δv formulation (commanded velocities) with KF/EKF.
#
# Deterministic vs stochastic simulation:
#   - deterministic: no process noise, no measurement noise
#   - stochastic:    process noise on plant pose + measurement noise on position
#
# Estimation + certainty equivalence (matches the lecture slides):
#   1) Measure y(k)
#   2) KF/EKF update -> \hat x(k|k)
#   3) Solve MPC using \hat x(k|k) as initial condition
#   4) Apply u(k)=u^*(k|k) and simulate plant forward
#
# Loads either (controller):
#   - qp_OCP_dv.casadi                      (linear lifted QP with Δv)
#   - nlp_circle_wall_obstacle_dv.casadi    (nonlinear multiple-shooting NLP with Δv)
#
# Loads either (estimator):
#   - linear_kf_step.casadi                 (general linear KF step)
#   - ekf_step_nonlinear_dv.casadi          (EKF step for nonlinear Δv model)
#

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def build_circle_reference(center, radius, w, t_k, dt, H):
    nz = 2
    Gamma = np.zeros((nz * H, 1))
    for j in range(1, H + 1):
        t = t_k + j * dt
        rj = center + radius * np.array([np.cos(w * t), np.sin(w * t)])
        Gamma[(j - 1) * nz : j * nz, 0] = rj
    return Gamma


def wrapToPi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


def pose_rhs(xp, v_cmd):
    psi = xp[2]
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array(
        [
            c * v_cmd[0] - s * v_cmd[1],
            s * v_cmd[0] + c * v_cmd[1],
            v_cmd[2],
        ],
        dtype=float,
    )


def rk4_step_pose(xp, v_cmd, dt):
    k1 = pose_rhs(xp, v_cmd)
    k2 = pose_rhs(xp + 0.5 * dt * k1, v_cmd)
    k3 = pose_rhs(xp + 0.5 * dt * k2, v_cmd)
    k4 = pose_rhs(xp + dt * k3, v_cmd)
    xp_next = xp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    xp_next[2] = wrapToPi(xp_next[2])
    return xp_next


def main():
    plt.ion()

    # %% ------------------------------- User options ----------------------------
    dt_mpc = 0.1      # MUST match dt used when exporting the solver
    H      = 20       # MUST match horizon used when exporting the solver

    dt_sim = 0.01     # plant simulation step (faster than dt_mpc)
    N_sub_exact = dt_mpc / dt_sim
    if abs(N_sub_exact - round(N_sub_exact)) > 1e-12:
        raise ValueError("Choose dt_sim so that dt_mpc/dt_sim is an integer.")
    N_sub = int(round(N_sub_exact))

    T_sim = 20                        # seconds
    N_mpc = int(round(T_sim / dt_mpc))

    use_nlp = True        # true: nonlinear MPC (NLP) + EKF, false: linear MPC (QP) + KF
    use_stochastic = True  # true: process + measurement noise, false: deterministic

    anim_pause = 0.00      # pause time for animation (0 = as fast as possible)

    # Controller solver files (change only these strings if you want other versions of your solvers)
    qp_solver_file  = 'qp_OCP_dv.casadi'                    # e.g. 'qp_OCP_dv_slack.casadi'
    nlp_solver_file = 'nlp_circle_wall_obstacle_dv.casadi'  # e.g. 'nlp_circle_wall_obstacle_dv_slack.casadi'

    np.random.seed(1)

    # %% --------------------------- Problem parameters --------------------------
    radius = 4.0
    center = np.array([0.0, 0.0], dtype=float)
    w      = 2 * 2 * np.pi / (200 * dt_mpc)

    b_wall = 0.0

    p_obs  = np.array([4.0, b_wall], dtype=float)
    r_obs  = 0.55
    r_safe = r_obs

    # Bounds (for plotting / saturation; must match the OCP you exported)
    v_max     = 3.0
    omega_max = 3.0

    dv_max     = 2.0 * dt_mpc
    domega_max = 3.0 * dt_mpc

    # %% --------------------------- Noise parameters ----------------------------
    # Measurement: y(k) = [x1(k); x2(k)] + v(k)
    sigma_y = 0.01                         # [m]
    R_meas  = (sigma_y**2) * np.eye(2)
    Lr      = np.linalg.cholesky(R_meas)

    # Process: additive noise on the *plant pose* at the sampling instant
    #   xp_true(k+1) = xp_true_nom(k+1) + w_p(k)
    sigma_p_proc   = 0.01                  # [m] per dt_mpc step
    sigma_psi_proc = 0.01                  # [rad] per dt_mpc step

    # %% ------------------------ Load solvers and estimators --------------------
    if use_nlp:
        solve = ca.Function.load(nlp_solver_file)
        nx_ocp = 6
        nu_ocp = 3
        nz = 2
        ekf_step = ca.Function.load('ekf_step_nonlinear_dv.casadi')
    else:
        solve = ca.Function.load(qp_solver_file)
        nx_ocp = 4
        nu_ocp = 2
        nz = 2
        kf_step = ca.Function.load('linear_kf_step.casadi')

    # %% ------------------------------ Initial condition ------------------------
    # True plant pose: [x1; x2; psi]
    xp_true = np.array([radius + 2.0, b_wall + 2.0, 0.0], dtype=float)

    # Previously applied commanded velocity v(k-1) (body-frame velocities and yaw-rate)
    v_prev_cmd = np.array([0.0, 0.0, 0.0], dtype=float)   # [v1; v2; omega]

    # Previously applied Δv input (used by KF/EKF at the next measurement update)
    if use_nlp:
        u_applied = np.zeros((3, 1))        # [dv1; dv2; domega]
    else:
        u_applied = np.zeros((2, 1))        # [dv1; dv2]

    # Initial measurement
    if use_stochastic:
        y_meas = xp_true[0:2].reshape(2, 1) + Lr @ np.random.randn(2, 1)
    else:
        y_meas = xp_true[0:2].reshape(2, 1)

    # Initial estimate and covariance
    if use_nlp:
        xhat = np.vstack([y_meas, [[0.0], [0.0], [0.0], [0.0]]])     # [x1;x2;psi; v1_prev; v2_prev; omega_prev]
        P    = np.diag([0.2, 0.2, 0.2, 0.5, 0.5, 0.5])
        # Filter design covariance (not necessarily equal to plant noise)
        Qw   = np.diag([sigma_p_proc**2, sigma_p_proc**2, sigma_psi_proc**2, 1e-4, 1e-4, 1e-4])
    else:
        xhat = np.vstack([y_meas, [[0.0], [0.0]]])               # [x1;x2; v1_prev; v2_prev]
        P    = np.diag([0.2, 0.2, 0.5, 0.5])
        Qw   = np.diag([sigma_p_proc**2, sigma_p_proc**2, 1e-3, 1e-3])

    # %% ------------------------------ Logging ----------------------------------
    N_log = N_mpc * N_sub + 1
    t_log    = np.zeros((N_log, 1))
    xp_log   = np.zeros((3, N_log))
    xhat_log = np.zeros((xhat.shape[0], N_log))

    t_log[0, 0]    = 0.0
    xp_log[:, 0]   = xp_true
    xhat_log[:, 0] = xhat[:, 0]

    psi_hat_plot = float(xp_true[2])
    if use_nlp:
        psi_hat_plot = float(xhat[2, 0])

    # %% ========================================================================
    #                        Animation setup (MPC-level)
    # ========================================================================
    tt_dense = np.linspace(0.0, T_sim, 1000)
    circ = center.reshape(2, 1) + radius * np.vstack((np.cos(w * tt_dense), np.sin(w * tt_dense)))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25)

    # --- XY plot (large)
    ax_xy = fig.add_subplot(gs[0:2, 0:3])
    ax_xy.grid(True)
    ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.set_xlabel(r"$x_1$")
    ax_xy.set_ylabel(r"$x_2$")
    ax_xy.set_title(r"MPC + KF/EKF ($\Delta v$)")

    h_refdense, = ax_xy.plot(circ[0, :], circ[1, :], "--", linewidth=1.2)
    xx = np.linspace(center[0] - radius - 4, center[0] + radius + 4, 2)
    h_wall, = ax_xy.plot(xx, b_wall * np.ones_like(xx), "k-", linewidth=2)

    if use_nlp:
        th = np.linspace(0.0, 2.0 * np.pi, 240)
        h_obs, = ax_xy.plot(p_obs[0] + r_safe * np.cos(th), p_obs[1] + r_safe * np.sin(th), "k-", linewidth=2)

    ax_xy.set_xlim([center[0] - radius - 3.0, center[0] + radius + 3.0])
    ax_xy.set_ylim([b_wall - 0.8, center[1] + radius + 3.0])

    h_cl_true, = ax_xy.plot([xp_true[0]], [xp_true[1]], "-", linewidth=2)
    h_cl_hat,  = ax_xy.plot([xhat[0, 0]], [xhat[1, 0]], "-.", linewidth=2)

    h_pt_true, = ax_xy.plot([xp_true[0]], [xp_true[1]], "o", markersize=7)
    h_pt_hat,  = ax_xy.plot([xhat[0, 0]], [xhat[1, 0]], "x", markersize=9, linewidth=2)

    Lh = 0.6
    h_head_true, = ax_xy.plot(
        [xp_true[0], xp_true[0] + Lh * np.cos(xp_true[2])],
        [xp_true[1], xp_true[1] + Lh * np.sin(xp_true[2])],
        "r-",
        linewidth=2,
    )
    h_head_hat, = ax_xy.plot(
        [xhat[0, 0], xhat[0, 0] + Lh * np.cos(psi_hat_plot)],
        [xhat[1, 0], xhat[1, 0] + Lh * np.sin(psi_hat_plot)],
        "b-",
        linewidth=2,
    )

    h_pred, = ax_xy.plot([np.nan], [np.nan], "-", linewidth=2)
    h_ref,  = ax_xy.plot([np.nan], [np.nan], ":", linewidth=2)

    if use_nlp:
        ax_xy.legend(
            [h_refdense, h_wall, h_obs, h_cl_true, h_cl_hat, h_pt_true, h_pt_hat, h_pred, h_ref],
            ["reference (dense)", r"wall $x_2\geq b_{\rm wall}$", "obstacle",
             "true trajectory", "estimated trajectory", "true state", "estimated state",
             "predicted", "reference segment"],
            loc="best",
        )
    else:
        ax_xy.legend(
            [h_refdense, h_wall, h_cl_true, h_cl_hat, h_pt_true, h_pt_hat, h_pred, h_ref],
            ["reference (dense)", r"wall $x_2\geq b_{\rm wall}$",
             "true trajectory", "estimated trajectory", "true state", "estimated state",
             "predicted", "reference segment"],
            loc="best",
        )

    # Time axis placeholder
    t_u0 = (np.arange(H) * dt_mpc).astype(float)

    # --- Δv plots
    ax_dv1 = fig.add_subplot(gs[2, 0]); ax_dv1.grid(True)
    ax_dv1.set_title(r"$\Delta v_1(\cdot|k)$"); ax_dv1.set_xlabel("time [s]"); ax_dv1.set_ylabel("increment")
    h_dv1, = ax_dv1.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_dv1.axhline(-dv_max, linestyle="--", color="k"); ax_dv1.axhline(dv_max, linestyle="--", color="k")

    ax_dv2 = fig.add_subplot(gs[2, 1]); ax_dv2.grid(True)
    ax_dv2.set_title(r"$\Delta v_2(\cdot|k)$"); ax_dv2.set_xlabel("time [s]"); ax_dv2.set_ylabel("increment")
    h_dv2, = ax_dv2.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_dv2.axhline(-dv_max, linestyle="--", color="k"); ax_dv2.axhline(dv_max, linestyle="--", color="k")

    ax_dw = fig.add_subplot(gs[2, 2]); ax_dw.grid(True)
    ax_dw.set_title(r"$\Delta \omega(\cdot|k)$"); ax_dw.set_xlabel("time [s]"); ax_dw.set_ylabel("increment")
    h_dw, = ax_dw.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_dw.axhline(-domega_max, linestyle="--", color="k"); ax_dw.axhline(domega_max, linestyle="--", color="k")
    if not use_nlp:
        ax_dw.text(0.05, 0.85, r"QP: $\Delta\omega\equiv 0$", transform=ax_dw.transAxes)

    # --- v plots
    ax_v1 = fig.add_subplot(gs[3, 0]); ax_v1.grid(True)
    ax_v1.set_title(r"$v_1(\cdot|k)$"); ax_v1.set_xlabel("time [s]"); ax_v1.set_ylabel("vel")
    h_v1, = ax_v1.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_v1.axhline(-v_max, linestyle="--", color="k"); ax_v1.axhline(v_max, linestyle="--", color="k")

    ax_v2 = fig.add_subplot(gs[3, 1]); ax_v2.grid(True)
    ax_v2.set_title(r"$v_2(\cdot|k)$"); ax_v2.set_xlabel("time [s]"); ax_v2.set_ylabel("vel")
    h_v2, = ax_v2.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_v2.axhline(-v_max, linestyle="--", color="k"); ax_v2.axhline(v_max, linestyle="--", color="k")

    ax_om = fig.add_subplot(gs[3, 2]); ax_om.grid(True)
    ax_om.set_title(r"$\omega(\cdot|k)$"); ax_om.set_xlabel("time [s]"); ax_om.set_ylabel("yaw rate")
    h_om, = ax_om.plot(t_u0, np.zeros_like(t_u0), "-", linewidth=1.6)
    ax_om.axhline(-omega_max, linestyle="--", color="k"); ax_om.axhline(omega_max, linestyle="--", color="k")
    if not use_nlp:
        ax_om.text(0.05, 0.85, r"QP: $\omega\equiv 0$", transform=ax_om.transAxes)

    plt.show(block=False)
    plt.pause(0.05)

    # %% ------------------------------ Warm-start guesses -----------------------
    if use_nlp:
        Xguess = np.zeros((nx_ocp, H + 1))
        Uguess = np.zeros((nu_ocp, H))
        Xguess[:, [0]] = xhat
    else:
        Uguess = np.zeros((nu_ocp * H, 1))

    # %% ========================================================================
    #                              MPC LOOP
    # ========================================================================
    idx = 1

    for k in range(N_mpc):
        t_k = k * dt_mpc

        # (1) OPTIMIZE: solve MPC using the current estimate \hat x(k|k)
        Gamma_k = build_circle_reference(center, radius, w, t_k, dt_mpc, H)
        ref_seg = Gamma_k.reshape(H, nz).T

        if use_nlp:
            # x0_ocp should be the full augmented estimate [xp(k); v(k-1)]
            x0_ocp = xhat

            U_sol, X_sol, J_sol = solve(
                ca.DM(x0_ocp),
                ca.DM(Gamma_k),
                ca.DM([[b_wall]]),
                ca.DM(Xguess),
                ca.DM(Uguess),
            )
            U_opt = np.array(U_sol.full())
            X_opt = np.array(X_sol.full())
            J_opt = float(J_sol)

            # Extract first increment dv0 = Δv(k|k) and compute commanded velocity:
            dv0   = U_opt[:, [0]]
            v_cmd = xhat[3:6, :] + dv0

            x1_pred = X_opt[0, :]
            x2_pred = X_opt[1, :]

            dv1_seq = U_opt[0, :]
            dv2_seq = U_opt[1, :]
            dw_seq  = U_opt[2, :]

            v1_seq = X_opt[3, 1:]
            v2_seq = X_opt[4, 1:]
            om_seq = X_opt[5, 1:]

            u_applied = dv0   # applied over [k,k+1)

        else:
            # x0_ocp should be the linear augmented estimate [p(k); v(k-1)]
            x0_ocp = xhat

            U_sol, X_sol, J_sol = solve(
                ca.DM(x0_ocp),
                ca.DM(Gamma_k),
                ca.DM([[b_wall]]),
                ca.DM(Uguess),
            )
            U_opt_vec = np.array(U_sol.full())
            X_opt_vec = np.array(X_sol.full())
            J_opt     = float(J_sol)

            U_opt = U_opt_vec.reshape(nu_ocp, H, order="F")
            X_mat = X_opt_vec.reshape(nx_ocp, H, order="F")

            # Extract first increment dv0 = Δv(k|k) and compute commanded velocity:
            dv0   = U_opt[:, [0]]
            v_cmd = np.vstack([xhat[2:4, :] + dv0, [[0.0]]])

            x1_pred = np.concatenate([[xhat[0, 0]], X_mat[0, :]])
            x2_pred = np.concatenate([[xhat[1, 0]], X_mat[1, :]])

            dv1_seq = U_opt[0, :]
            dv2_seq = U_opt[1, :]
            dw_seq  = np.zeros((H,))

            v1_seq = X_mat[2, :]
            v2_seq = X_mat[3, :]
            om_seq = np.zeros((H,))

            u_applied = dv0   # applied over [k,k+1)

        # Saturate commanded velocity
        v_cmd[0, 0] = float(np.clip(v_cmd[0, 0], -v_max, v_max))
        v_cmd[1, 0] = float(np.clip(v_cmd[1, 0], -v_max, v_max))
        v_cmd[2, 0] = float(np.clip(v_cmd[2, 0], -omega_max, omega_max))

        # Animation update (predicted trajectory + reference segment)
        h_pred.set_data(x1_pred, x2_pred)
        h_ref.set_data(ref_seg[0, :], ref_seg[1, :])

        t_u = t_k + np.arange(H) * dt_mpc

        h_dv1.set_data(t_u, dv1_seq)
        h_dv2.set_data(t_u, dv2_seq)
        h_dw.set_data(t_u,  dw_seq)

        h_v1.set_data(t_u, v1_seq)
        h_v2.set_data(t_u, v2_seq)
        h_om.set_data(t_u, om_seq)

        ax_xy.set_title(f"k={k} (t={t_k:.2f} s),  J^*={J_opt:.2e}")
        for ax in [ax_dv1, ax_dv2, ax_dw, ax_v1, ax_v2, ax_om]:
            ax.set_xlim([t_k, t_k + (H - 1) * dt_mpc])

        plt.pause(max(anim_pause, 0.001))

        # (2) APPLY + SIMULATE plant over one MPC interval [k,k+1)
        v_cmd_flat = v_cmd[:, 0].astype(float)
        for j in range(1, N_sub + 1):
            xp_true = rk4_step_pose(xp_true, v_cmd_flat, dt_sim)

            idx += 1
            t_log[idx - 1, 0]    = t_k + j * dt_sim
            xp_log[:, idx - 1]   = xp_true
            xhat_log[:, idx - 1] = xhat[:, 0]

        if use_stochastic:
            # Discrete-time process noise applied at the sampling instant k+1 (affects heading)
            xp_true = xp_true + np.array(
                [
                    sigma_p_proc * np.random.randn(),
                    sigma_p_proc * np.random.randn(),
                    sigma_psi_proc * np.random.randn(),
                ],
                dtype=float,
            )
            xp_true[2] = wrapToPi(xp_true[2])

        # (3) MEASURE and (4) ESTIMATE at time k+1
        if use_stochastic:
            y_meas_next = xp_true[0:2].reshape(2, 1) + Lr @ np.random.randn(2, 1)
        else:
            y_meas_next = xp_true[0:2].reshape(2, 1)

        if use_stochastic:
            if use_nlp:
                # EKF step that predicts with u_applied and updates with y(k+1):
                #   [xhat_next, P_next] = ekf_step(xhat, P, u_applied, y_meas_next, dt_mpc, Qw, R_meas);
                xhat_next, P_next, *_ = ekf_step(
                    ca.DM(xhat),
                    ca.DM(P),
                    ca.DM(u_applied),
                    ca.DM(y_meas_next),
                    ca.DM([[dt_mpc]]),
                    ca.DM(Qw),
                    ca.DM(R_meas),
                )
                xhat = np.array(xhat_next.full())
                P    = np.array(P_next.full())
            else:
                A = np.array(
                    [
                        [1, 0, dt_mpc, 0],
                        [0, 1, 0, dt_mpc],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
                B = np.array(
                    [
                        [dt_mpc, 0],
                        [0, dt_mpc],
                        [1, 0],
                        [0, 1],
                    ],
                    dtype=float,
                )
                C = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ],
                    dtype=float,
                )

                # KF step that predicts with u_applied and updates with y(k+1):
                #   [xhat_next, P_next] = kf_step(xhat, P, u_applied, y_meas_next, A, B, C, Qw, R_meas);
                xhat_next, P_next, *_ = kf_step(
                    ca.DM(xhat),
                    ca.DM(P),
                    ca.DM(u_applied),
                    ca.DM(y_meas_next),
                    ca.DM(A),
                    ca.DM(B),
                    ca.DM(C),
                    ca.DM(Qw),
                    ca.DM(R_meas),
                )
                xhat = np.array(xhat_next.full())
                P    = np.array(P_next.full())
        else:
            if use_nlp:
                xhat = np.vstack([xp_true.reshape(3, 1), v_cmd.reshape(3, 1)])
            else:
                xhat = np.vstack([xp_true[0:2].reshape(2, 1), v_cmd[0:2].reshape(2, 1)])

        # Update plots with true + estimated state at time k+1
        h_cl_true.set_data(xp_log[0, :idx], xp_log[1, :idx])
        h_cl_hat.set_data(xhat_log[0, :idx], xhat_log[1, :idx])
        h_pt_true.set_data([xp_true[0]], [xp_true[1]])
        h_pt_hat.set_data([xhat[0, 0]], [xhat[1, 0]])

        if use_nlp:
            psi_hat_plot = float(xhat[2, 0])
        else:
            if use_stochastic:
                psi_hat_plot = wrapToPi(psi_hat_plot + float(v_cmd[2, 0]) * dt_mpc)
            else:
                psi_hat_plot = float(xp_true[2])

        h_head_true.set_data(
            [xp_true[0], xp_true[0] + Lh * np.cos(xp_true[2])],
            [xp_true[1], xp_true[1] + Lh * np.sin(xp_true[2])],
        )
        h_head_hat.set_data(
            [xhat[0, 0], xhat[0, 0] + Lh * np.cos(psi_hat_plot)],
            [xhat[1, 0], xhat[1, 0] + Lh * np.sin(psi_hat_plot)],
        )

        plt.pause(max(anim_pause, 0.001))

        # Warm start for next MPC solve (k <- k+1)
        if use_nlp:
            Uguess = np.hstack([U_opt[:, 1:], U_opt[:, [-1]]])
            Xguess = np.hstack([X_opt[:, 1:], X_opt[:, [-1]]])
            Xguess[:, [0]] = xhat
        else:
            Uguess = np.vstack([U_opt_vec[nu_ocp:, :], np.zeros((nu_ocp, 1))])

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()