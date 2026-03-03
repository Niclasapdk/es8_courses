# Builds a general discrete-time linear Kalman Filter step as a CasADi Function and saves it.
#
# Usage:
#   python make_linear_kf_casadi_filled.py
# -> writes: linear_kf_step.casadi
#
# The exported function is:
#   (xhat, P, u, y, A, B, C, Qw, Rv) -> (xhat_upd, P_upd, K, x_pred, P_pred, innov)

import casadi as ca


def main():
    # Dimensions (the exported function is sized by these)
    nx = 4   # state dimension
    nu = 2   # input dimension
    ny = 2   # measurement dimension

    # Symbols
    xhat = ca.MX.sym('xhat', nx, 1)
    P    = ca.MX.sym('P',    nx, nx)
    u    = ca.MX.sym('u',    nu, 1)
    y    = ca.MX.sym('y',    ny, 1)

    A    = ca.MX.sym('A',    nx, nx)
    B    = ca.MX.sym('B',    nx, nu)
    C    = ca.MX.sym('C',    ny, nx)
    Qw   = ca.MX.sym('Qw',   nx, nx)  # process noise covariance
    Rv   = ca.MX.sym('Rv',   ny, ny)  # measurement noise covariance

    I = ca.MX.eye(nx)

    # KF prediction
    x_pred = A @ xhat + B @ u
    P_pred = A @ P @ A.T + Qw

    # KF update
    S = C @ P_pred @ C.T + Rv
    K = P_pred @ C.T @ ca.inv(S)

    innov = y - C @ x_pred
    x_upd = x_pred + K @ innov

    # Joseph form covariance update (numerically robust)
    P_upd = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ Rv @ K.T

    kf_step = ca.Function(
        'kf_step',
        [xhat, P, u, y, A, B, C, Qw, Rv],
        [x_upd, P_upd, K, x_pred, P_pred, innov],
        ['xhat', 'P', 'u', 'y', 'A', 'B', 'C', 'Qw', 'Rv'],
        ['xhat_upd', 'P_upd', 'K', 'x_pred', 'P_pred', 'innov'],
    )

    kf_step.save('linear_kf_step.casadi')
    print('Saved: linear_kf_step.casadi')


if __name__ == "__main__":
    main()
