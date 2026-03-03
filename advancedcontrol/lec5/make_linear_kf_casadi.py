# Builds a general discrete-time linear Kalman Filter step as a CasADi Function and saves it.

import casadi as ca


def main():
    # Dimensions (example defaults; the exported function is sized by these)
    nx = 4   # state dimension
    nu = 2   # input dimension
    ny = 2   # measurement dimension
    # Identity matrix in casadi to help later:     I = ca.MX.eye(nx)

    # Symbols
    xhat = ca.MX.sym('xhat', nx, 1) 
    P    = ca.MX.sym('P',    nx, nx) #Covariance matrix
    u    = ca.MX.sym('u',    nu, 1) #Input
    y    = ca.MX.sym('y',    ny, 1) #output

    A    = ca.MX.sym('A',    nx, nx) #State matrix 
    B    = ca.MX.sym('B',    nx, nu) # Input matrix
    C    = ca.MX.sym('C',    ny, nx) # Output matrix
    Qw   = ca.MX.sym('Qw',   nx, nx) #Process noise covariance (Here, E=I)
    Rv   = ca.MX.sym('Rv',   ny, ny) #Measurement noise covariance

    # KF prediction
    x_pred = # Fill in
    P_pred = # Fill in

    # KF update
    S = C @ P_pred @ C.T + Rv
    K = #Fill in (Kalman gain)

    innov = y - C @ x_pred
    x_upd = #Fill in

    # Joseph form
    P_upd = #Fill in (the identity matrix I = ca.MX.eye(nx) can be helpful)

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