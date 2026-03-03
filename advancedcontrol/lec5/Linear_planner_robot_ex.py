# Linear_planner_robot_ex_filled.py
#
# Filled version of the exercise template Linear_planner_robot_ex.py (Δv formulation).
# IMPORTANT for MPCrun.py:
#   - dt and H MUST match dt_mpc and H inside MPCrun.py when you export qp_OCP_dv.casadi.
#
# Exports: qp_OCP_dv.casadi

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def build_lifting_AB(A: np.ndarray, B: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    nx = A.shape[0]
    nu = B.shape[1]

    Acal = np.zeros((nx * H, nx))
    Bcal = np.zeros((nx * H, nu * H))

    Apow = [np.eye(nx)]
    for _ in range(1, H + 1):
        Apow.append(A @ Apow[-1])  # A^k

    for k in range(1, H + 1):
        Acal[(k - 1) * nx : k * nx, :] = Apow[k]  # A^k
        for j in range(1, k + 1):
            Akj = Apow[k - j]  # A^(k-j)
            Bcal[(k - 1) * nx : k * nx, (j - 1) * nu : j * nu] = Akj @ B

    return Acal, Bcal


def main():
    dt = 0.1  # Sampling time
    H = 20    # horizon  (set to 20 to match MPCrun.py defaults)

    # Augmented state x = [pos1; pos2; v1_prev; v2_prev], input u = Δv
    nx = 4
    nu = 2
    nz = 2  # track position z=[x1;x2]

    # Wall constraint:
    b_wall = 0.0

    # Bounds (match MPCrun.py defaults)
    u_max = 2.0 * dt   # max componentwise Δv
    v_max = 3.0        # max componentwise commanded velocity

    # Weights (reasonable defaults; tune in the exercise)
    Q = np.diag([10.0, 10.0])
    R = 0.2 * np.eye(nu)

    # Initial augmented state:
    radius = 4.0
    center = np.array([0.0, 0.0])
    x0_val = np.array([radius + 2.0, b_wall + 2.0, 0.0, 0.0])

    # Reference (circle) sampled at k=1,...,H
    w = 2 * 2 * np.pi / (200 * dt)
    Gamma_val = np.zeros((nz * H, 1))
    for k in range(1, H + 1):
        t = k * dt
        rk = center + radius * np.array([np.cos(w * t), np.sin(w * t)])
        Gamma_val[(k - 1) * nz : k * nz, 0] = rk

    # Discrete-time model:
    # v(k)   = v(k-1) + Δv(k)
    # p(k+1) = p(k) + dt * v(k)
    # with x(k) = [p(k); v(k-1)], u(k)=Δv(k)
    A = np.array(
        [
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [dt, 0.0],
            [0.0, dt],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    Cz = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    # Lifting matrices  X = Acal*x0 + Bcal*U
    Acal, Bcal = build_lifting_AB(A, B, H)

    # Lift Cz, Q, R (block diagonal)
    Czcal = ca.DM(np.kron(np.eye(H), Cz))
    Qcal  = ca.DM(np.kron(np.eye(H), Q))
    Rcal  = ca.DM(np.kron(np.eye(H), R))

    # Selection matrix for wall constraint: pick x2(k) from stacked X
    sx  = np.array([[0.0, 1.0, 0.0, 0.0]])
    Sx2 = ca.DM(np.kron(np.eye(H), sx))

    # Selection matrix for velocity constraints: pick [x3; x4] from stacked X
    sv = np.array([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    Sv = ca.DM(np.kron(np.eye(H), sv))

    # CasADi Opti QP
    opti = ca.Opti("conic")

    x0    = opti.parameter(nx, 1)
    Gamma = opti.parameter(nz * H, 1)
    bw    = opti.parameter(1, 1)

    # Decision variable: stacked Δv inputs U=[u(0);...;u(H-1)]
    U = opti.variable(nu * H, 1)

    X = ca.DM(Acal) @ x0 + ca.DM(Bcal) @ U      # stacked states: [x(1);...;x(H)]
    E = Gamma - Czcal @ X                        # stacked tracking error

    J = ca.mtimes([E.T, Qcal, E]) + ca.mtimes([U.T, Rcal, U])
    opti.minimize(J)

    # Δv bounds
    opti.subject_to(-u_max * ca.DM.ones(nu * H, 1) <= U)
    opti.subject_to(U <= u_max * ca.DM.ones(nu * H, 1))

    # Wall constraint: x2(k) >= b_wall for k=1..H
    opti.subject_to(Sx2 @ X >= bw * ca.DM.ones(H, 1))

    # Velocity constraint on predicted velocities
    opti.subject_to(Sv @ X <=  v_max * ca.DM.ones(2 * H, 1))
    opti.subject_to(Sv @ X >= -v_max * ca.DM.ones(2 * H, 1))

    opti.solver("qpoases")

    # Export solver
    solve_qp = opti.to_function(
        "solve_qp",
        [x0, Gamma, bw, U],
        [U, X, opti.f],
        ["x0", "Gamma", "b_wall", "Uguess"],
        ["U_opt", "X_opt", "J_opt"],
    )
    solve_qp.save("qp_OCP_dv.casadi")
    print("Saved: qp_OCP_dv.casadi")

    # Solve once (demo)
    solve_qp = ca.Function.load("qp_OCP_dv.casadi")

    Uguess = np.zeros((nu * H, 1))
    U_sol, X_sol, J_sol = solve_qp(x0_val.reshape(-1, 1), Gamma_val, np.array([[b_wall]]), Uguess)

    X_opt = np.array(X_sol.full()).reshape(H, nx).T
    x1 = np.concatenate(([x0_val[0]], X_opt[0, :]))
    x2 = np.concatenate(([x0_val[1]], X_opt[1, :]))

    print(f"QP solved. J = {float(J_sol):.6e}")

    # Plot
    plt.figure(figsize=(7, 7))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    tt = np.linspace(0, H * dt, 400)
    circ = center.reshape(2, 1) + radius * np.vstack((np.cos(w * tt), np.sin(w * tt)))
    plt.plot(circ[0, :], circ[1, :], "--", linewidth=1.2)

    xx = np.array([min(np.min(x1), np.min(circ[0, :])) - 2, max(np.max(x1), np.max(circ[0, :])) + 2])
    plt.plot(xx, b_wall * np.ones_like(xx), "k-", linewidth=2)

    plt.plot(x1, x2, "-", linewidth=2)
    plt.legend(["reference (circle)", r"wall $x_2\geq b_{\rm wall}$", "robot trajectory"], loc="best")
    plt.show()


if __name__ == "__main__":
    main()
