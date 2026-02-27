# planner_robot_qp_lifting_exercise.py
#
# (Exercise template; fill in the marked lines.)

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
    dt = 0.1  #Sampling time
    H = 200   #horizon

    #We will use the model where we optimize over acceleration even though we
    #command velocity to the robot (to ensure smooth velocities). 
    #We would like for the position of the robot to follow a specific. 
    #Specifically:
    #min \sum^{H-1}_{k=0} (r(k)-z(k))^T Q (r(k)-z(k)) + u(k)^T R u(k)
    #reference. What do you think z should be in this case if x and u are: 
    nx = 4   # x = [x1; x2; x3; x4] = [pos1; pos2; v1; v2]
    nu = 2   # u = [u1; u2] = [a1; a2]
    nz =  #Fill in;   # z = ???

    #We have a wall constraint:
    b_wall = 0.0

    #Bounds on the acceleration:
    u_min = np.array([-2.0, -2.0])
    u_max = np.array([ 2.0,  2.0])

    Q =  #Fill in
    R =  #Fill in

    # Start outside the circle (and feasible wrt wall)
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

    ## Discrete-time model  x(k+1) = A x(k) + B u(k)  and z(k)=Cz x(k)
    A =  #Fill in
    B =  #Fill in
    Cz =  #Fill in

    ## Lifting matrices  X = Acal*x0 + Bcal*U
    Acal, Bcal = build_lifting_AB(A, B, H)

    #Remember, you need to also lift Q, R, and C_z (blockdiagonal). 
    #Easy way in python to do blockdiagonal with repeated matrices MAT is to use
    #np.kron(np.eye(H),MAT) (if we want to repeat MAT H times)
    Czcal =  #Fill in
    Qcal  =  #Fill in
    Rcal  =  #Fill in

    #Note: we can also make a matrix similar to C_z which picks x_2 for the
    #wall constraint if you want.
    #Sx2 = Fill in;  % picks x2(k), k=1..H from stacked X
    Sx2 =  # Fill in  (picks x2(k), k=1..H from stacked X)

    Umin = np.kron(np.ones((H, 1)), u_min.reshape(-1, 1))
    Umax = np.kron(np.ones((H, 1)), u_max.reshape(-1, 1))

    ## CasADi Opti QP
    #conic tells it that we want to solve a QP:
    opti = ca.Opti("conic")
    #We want to make a function such that I give it x0, Gamma, bw, and it
    #returns to me the optimal solution U, X, and the value of the cost.

    #Parameters (inputs to the optimal control problem function):
    x0 = opti.parameter(nx, 1)
    Gamma = opti.parameter(nz * H, 1)
    bw = opti.parameter(1, 1)

    #The lifted input! since python starts from 0 indexing, U contains
    #[u(0); u(1); ...; u(H-1)] stacked in a single vector.
    U = opti.variable(nu * H, 1)

    X = Acal @ x0 + Bcal @ U              # stacked states: [x(1);...;x(H)]
    E = Gamma - Czcal @ X                 # stacked tracking error

    J =  #Fill in       # QP objective
    opti.minimize(J)

    # Input bounds:
    #Opti understands that it is element-wise. No need to lift constraints
    #explicitly.
    opti.subject_to(Umin <= U)
    opti.subject_to(U <= Umax)

    # Wall constraint: x2(k) >= b_wall for k=1..H
    opti.subject_to(Sx2 @ X >= bw * ca.DM.ones(H, 1)) #Here I used a selection matrix 

    # qpOASES
    opti.solver("qpoases")

    ## Export solver map (optimal control function):
    # (x0, Gamma, b_wall, Uguess) -> (U*, X*, J*) 
    solve_qp = opti.to_function(
        "solve_qp",
        [x0, Gamma, bw, U],
        [U, X, opti.f],
        ["x0", "Gamma", "b_wall", "Uguess"],
        ["U_opt", "X_opt", "J_opt"],
    )
    solve_qp.save("qp_OCP.casadi")

    ## Solve once
    solve_qp = ca.Function.load("qp_OCP.casadi")

    Uguess = np.zeros((nu * H, 1)) #Initial guess to the solver
    U_sol, X_sol, J_sol = solve_qp(x0_val.reshape(-1, 1), Gamma_val, np.array([[b_wall]]), Uguess)
    U_opt = np.array(U_sol.full()).reshape(-1, 1)
    X_opt = np.array(X_sol.full()).reshape(-1, 1)
    J_opt = float(J_sol)

    print(f"QP solved. J = {J_opt:.6e}")

    # Unstack X to get x(1)...x(H)
    X_mat = X_opt.reshape(H, nx).T
    x1 = np.concatenate(([x0_val[0]], X_mat[0, :]))
    x2 = np.concatenate(([x0_val[1]], X_mat[1, :]))

    # Reference for plotting (include k=0 point for convenience)
    ref = Gamma_val.reshape(H, nz).T
    r1 = np.concatenate(([np.nan], ref[0, :]))  # no ref at k=0 in Gamma
    r2 = np.concatenate(([np.nan], ref[1, :]))

    ## Simple animation
    plt.figure(figsize=(7, 7))
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    # Reference circle (dense)
    tt = np.linspace(0, H * dt, 400)
    circ = center.reshape(2, 1) + radius * np.vstack((np.cos(w * tt), np.sin(w * tt)))
    plt.plot(circ[0, :], circ[1, :], "--", linewidth=1.2)

    # Wall line x2 = b_wall
    xx = np.array([min(np.min(x1), np.min(circ[0, :])) - 2, max(np.max(x1), np.max(circ[0, :])) + 2])
    plt.plot(xx, b_wall * np.ones_like(xx), "k-", linewidth=2)

    # Trajectory handles
    (h_traj,) = plt.plot([x1[0]], [x2[0]], "-", linewidth=2)
    (h_pt,) = plt.plot([x1[0]], [x2[0]], "o", markersize=8)

    plt.legend(
        ["reference (circle)", r"wall $x_2\geq b_{\rm wall}$", "robot trajectory"],
        loc="best",
    )

    for k in range(1, H + 2, 2):
        h_traj.set_data(x1[:k], x2[:k])
        h_pt.set_data([x1[k - 1]], [x2[k - 1]])
        plt.pause(0.05)

    plt.show()


if __name__ == "__main__":
    main()