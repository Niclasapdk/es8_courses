# Builds an EKF step for the nonlinear planar robot with heading under Δv inputs and saves it.

import casadi as ca


def main():
    nx = 6   # [x1;x2;psi; v1_prev; v2_prev; omega_prev]
    nu = 3   # [dv1; dv2; domega]
    ny = 2   # [x1;x2] measurement

    xhat = ca.MX.sym('xhat', nx, 1)
    P    = ca.MX.sym('P',    nx, nx)
    u    = ca.MX.sym('u',    nu, 1)
    y    = ca.MX.sym('y',    ny, 1)
    dt   = ca.MX.sym('dt',   1,  1)
    Qw   = ca.MX.sym('Qw',   nx, nx)
    Rv   = ca.MX.sym('Rv',   ny, ny)

    ## Pose kinematics: xp = [x1;x2;psi], commanded v = [v1;v2;omega]
    xp = ca.MX.sym('xp', 3, 1)
    vc = ca.MX.sym('vc', 3, 1)

    psi = xp[2]
    c = ca.cos(psi)
    s = ca.sin(psi)

    x1_dot  = c * vc[0] - s * vc[1]
    x2_dot  = s * vc[0] + c * vc[1]
    psi_dot = vc[2]

    f_ct = ca.Function('f_ct', [xp, vc], [ca.vertcat(x1_dot, x2_dot, psi_dot)])

    ## RK4 integrator for pose with ZOH on vc over dt
    k1 = f_ct(xp, vc)
    k2 = f_ct(xp + (dt / 2) * k1, vc)
    k3 = f_ct(xp + (dt / 2) * k2, vc)
    k4 = f_ct(xp + dt * k3, vc)
    xp_next_rk4 = xp + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    F_pose = ca.Function('F_pose', [xp, vc, dt], [xp_next_rk4])

    ## EKF prediction model (Δv augmentation):
    v_prev = xhat[3:6]
    v_cmd  = v_prev + u

    xp_k    = xhat[0:3]
    xp_pred = F_pose(xp_k, v_cmd, dt)

    psi_pred = xp_pred[2]
    psi_wrap = ca.atan2(ca.sin(psi_pred), ca.cos(psi_pred))
    xp_pred  = ca.vertcat(xp_pred[0], xp_pred[1], psi_wrap)

    x_pred = ca.vertcat(xp_pred, v_cmd)

    A = ca.jacobian(x_pred, xhat)

    ## Measurement model: y = [x1;x2] + noise  (H is constant)
    h = x_pred[0:2]
    C = ca.DM([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ])

    ## EKF covariance prediction
    P_pred = A @ P @ A.T + Qw

    ## EKF update
    S = C @ P_pred @ C.T + Rv
    K = P_pred @ C.T @ ca.inv(S)
    I = ca.MX.eye(nx)

    innov = y - h
    x_upd = x_pred + K @ innov

    psi_u = x_upd[2]
    x_upd_wrapped = ca.vertcat(
        x_upd[0],
        x_upd[1],
        ca.atan2(ca.sin(psi_u), ca.cos(psi_u)),
        x_upd[3],
        x_upd[4],
        x_upd[5],
    )

    P_upd = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ Rv @ K.T

    ekf_step = ca.Function(
        'ekf_step',
        [xhat, P, u, y, dt, Qw, Rv],
        [x_upd_wrapped, P_upd, K, x_pred, P_pred, innov],
        ['xhat', 'P', 'u', 'y', 'dt', 'Qw', 'Rv'],
        ['xhat_upd', 'P_upd', 'K', 'x_pred', 'P_pred', 'innov'],
    )

    ekf_step.save('ekf_step_nonlinear_dv.casadi')

    print('Saved: ekf_step_nonlinear_dv.casadi')


if __name__ == "__main__":
    main()