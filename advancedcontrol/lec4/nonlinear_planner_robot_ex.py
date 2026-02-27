# nonlinear_robot_circle_wall_obstacle_NLP_exercise.py
# Nonlinear planar robot with heading (multiple shooting, direct method).
#
# We plan with accelerations for smooth motion:
#   state  x = [x_1; x_2; psi; v_1; v_2; omega]
#          (position, heading, body-frame velocities, yaw rate)
#   input  u = [a_1; a_2; alpha]
#          (body-frame accelerations, yaw acceleration)
#
# Continuous-time dynamics:
#   [x_1dot; x_2dot] = R(psi)*[v_1; v_2]
#   psidot   = omega
#   v_1dot   = a_1
#   v_2dot   = a_2
#   omegadot = alpha
#
# We track a time-varying reference r(k) (circle) with constraints:
#   - Wall:      x_2(k) >= b_wall
#   - Obstacle:  (x_1(k)-x_obs)^2 + (x_2(k)-y_obs)^2 >= r_obs^2
#
# We build:
#   1) Continuous-time dynamics as a CasADi Function f_ct
#   2) Discrete-time integrator F_dt with Euler and RK4 options
#   3) Multiple-shooting NLP with CasADi Opti + IPOPT
#   4) Export an optimal control function with opti.to_function,
#      save it, load it, then solve using initial guesses for X and U
#   5) Animate (including heading)

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def main():
    ## Parameters
    dt = 0.1     #Sampling time
    H  = 200     #horizon

    use_rk4 = True   # true: RK4, false: Euler

    nx = 6   # x = [x1; x2; psi; v1; v2; omega]
    nu = 3   # u = [a1; a2; alpha]
    nz =  #FILL IN;   # z = ???

    # We have a wall constraint:
    b_wall = 0.0

    # Small circular obstacle on the wall
    p_obs  = np.array([4.0, b_wall])
    r_obs  = 0.55   # small obstacle radius
    r_safe = r_obs  #

    # Bounds on accelerations and angular acceleration:
    a_min     = np.array([-2.0, -2.0])
    a_max     = np.array([ 2.0,  2.0])
    alpha_min = -3.0
    alpha_max =  3.0

    # bounds on velocities and yaw-rate 
    v_max     = 3.0
    omega_max = 3.0

    # Tracking weights
    Q =  #Fill in
    R =  #Fill in   # penalize [a1; a2; alpha]
    Q_T =  #Fill in  # terminal tracking weight (optional)

    # Start outside the circle (and feasible wrt wall)
    radius = 4.0
    center = np.array([0.0, 0.0])
    x0_val = np.array([radius+2.0, b_wall+2.0, 0.0, 0.0, 0.0, 0.0])

    # Reference (circle) sampled at k=1,...,H
    w = 2*2*np.pi/(200*dt)
    Gamma_val = np.zeros((nz*H,1))
    for k in range(1, H+1):
        t = k*dt
        rk = center + radius*np.array([np.cos(w*t), np.sin(w*t)])
        Gamma_val[(k-1)*nz:k*nz, 0] = rk

    ## 1) Continuous-time dynamics  \dot{x} = f(x,u)  (CasADi Function)
    x = ca.MX.sym('x', nx, 1)
    u = ca.MX.sym('u', nu, 1)

    x1    = x[0]
    x2    = x[1]
    psi   = x[2]
    v1    = x[3]
    v2    = x[4]
    omega = x[5]

    a1    = u[0]
    a2    = u[1]
    alpha = u[2]

    c = ca.cos(psi)
    s = ca.sin(psi)

    x1_dot    = c*v1 - s*v2
    x2_dot    = s*v1 + c*v2
    psi_dot   = omega
    v1_dot    = a1
    v2_dot    = a2
    omega_dot = alpha

    xdot = ca.vertcat(x1_dot, x2_dot, psi_dot, v1_dot, v2_dot, omega_dot)
    f_ct = ca.Function('f_ct', [x,u], [xdot])

    ## 2) Discretized dynamics  x(k+1) = F_dt(x(k),u(k))  (CasADi Function)
    #    Here: Forward Euler and RK4 (both written, choose with use_rk4)
    dt_sym = ca.MX.sym('dt', 1, 1)

    # Euler
    F_euler = ca.Function('F_euler', [x,u,dt_sym], [x + dt_sym*f_ct(x,u)])

    # RK4
    k1 = f_ct(x,u)
    k2 = f_ct(x + (dt_sym/2)*k1, u)
    k3 = f_ct(x + (dt_sym/2)*k2, u)
    k4 = f_ct(x + dt_sym*k3, u)
    F_rk4 = ca.Function('F_rk4', [x,u,dt_sym], [x + (dt_sym/6)*(k1 + 2*k2 + 2*k3 + k4)])

    if use_rk4:
        F_dt = F_rk4
    else:
        F_dt = F_euler

    ## 3) Build OCP via lifting (Multiple Shooting) in CasADi Opti
    opti = ca.Opti('nlp')

    # Parameters (inputs to the optimal control problem function):
    x0    = opti.parameter(nx,1)
    Gamma = opti.parameter(nz*H,1)
    bw    = opti.parameter(1,1)

    # Decision variables (multiple shooting):
    X = opti.variable(nx, H+1)   # X(:,k+1) = x(k)
    U = opti.variable(nu, H)     # U(:,k)   = u(k-1) conceptually u(k-1)

    # Initial condition
    opti.subject_to(X[:,0] == x0)

    # Objective:
    #   \sum (r(k) - z(k))^T Q (r(k) - z(k)) + u(k)^T R u(k)

    # and Dynamics constraints (multiple shooting)
    for k in range(H):
        #FILL IN
        pass

    #Do not forget the terminal cost. 

    # Wall constraint: x2(k) >= b_wall
    #Fill in

    # Circular obstacle constraint
    #Fill in

    # Bounds on velocities and inputs:
    #Fill in

    opti.minimize(J)

    opti.solver('ipopt')

    ## Export solver map (optimal control function):
    # (x0, Gamma, b_wall, Xguess, Uguess) -> (U*, X*, J*)
    solve_nlp = opti.to_function(
        'solve_nlp',
        [x0, Gamma, bw, X, U],
        [U, X, opti.f],
        ['x0','Gamma','b_wall','Xguess','Uguess'],
        ['U_opt','X_opt','J_opt']
    )

    solve_nlp.save('nlp_circle_wall_obstacle.casadi')

    ## Solve once
    solve_nlp = ca.Function.load('nlp_circle_wall_obstacle.casadi')

    # Initial guesses for X and U (similar idea to Uguess in the QP)
    Xguess = np.zeros((nx, H+1))
    Uguess = np.zeros((nu, H))

    # Initialize positions with the reference, but enforce feasibility to the wall and obstacle
    Xguess[0,0] = x0_val[0]
    Xguess[1,0] = x0_val[1]
    Xguess[2,0] = x0_val[2]
    Xguess[3,0] = 0.0
    Xguess[4,0] = 0.0
    Xguess[5,0] = 0.0

    # Use reference for x(1..H) as a baseline guess
    ref = Gamma_val.reshape(H, nz).T
    Xguess[0,1:] = ref[0,:]
    Xguess[1,1:] = ref[1,:]

    # Project above the wall
    Xguess[1,:] = np.maximum(Xguess[1,:], b_wall + 0.15)

    # Add a small "bump" around the obstacle so the guess is feasible
    k_mid = int(round(H/2))
    kk = np.arange(H+1)
    bump  = np.exp(-((kk - k_mid)**2)/(2*(0.12*H)**2))
    Xguess[1,:] = Xguess[1,:] + (r_safe + 0.25)*bump

    # Solve
    U_sol, X_sol, J_sol = solve_nlp(
        ca.DM(x0_val.reshape(-1,1)),
        ca.DM(Gamma_val),
        ca.DM([[b_wall]]),
        ca.DM(Xguess),
        ca.DM(Uguess)
    )

    U_opt = np.array(U_sol.full())
    X_opt = np.array(X_sol.full())
    J_opt = float(J_sol)

    print(f"NLP solved. J = {J_opt:.6e}")

    ## Simple animation (includes heading)
    x1_traj  = X_opt[0,:]
    x2_traj  = X_opt[1,:]
    psi_traj = X_opt[2,:]

    plt.figure(figsize=(7,7))
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    # Reference circle (dense)
    tt = np.linspace(0, H*dt, 400)
    circ = center.reshape(2,1) + radius*np.vstack((np.cos(w*tt), np.sin(w*tt)))
    plt.plot(circ[0,:], circ[1,:], '--', linewidth=1.2)

    # Wall line x2 = b_wall
    xx = np.linspace(min(np.min(x1_traj), np.min(circ[0,:])) - 2,
                     max(np.max(x1_traj), np.max(circ[0,:])) + 2, 2)
    plt.plot(xx, b_wall*np.ones_like(xx), 'k-', linewidth=2)

    # Small circular obstacle
    th = np.linspace(0, 2*np.pi, 240)
    plt.plot(p_obs[0] + r_safe*np.cos(th), p_obs[1] + r_safe*np.sin(th), 'k-', linewidth=2)

    # Trajectory handles
    (h_traj,) = plt.plot([x1_traj[0]], [x2_traj[0]], '-', linewidth=2)
    (h_pt,)   = plt.plot([x1_traj[0]], [x2_traj[0]], 'o', markersize=8)

    # Heading arrow
    Lh = 0.6
    (h_head,) = plt.plot([x1_traj[0], x1_traj[0] + Lh*np.cos(psi_traj[0])],
                         [x2_traj[0], x2_traj[0] + Lh*np.sin(psi_traj[0])],
                         'r-', linewidth=2)

    plt.legend(
        ['reference (circle)', r'wall $x_2\geq b_{\rm wall}$', 'obstacle', 'robot trajectory'],
        loc='best'
    )

    for k in range(0, H+1, 2):
        h_traj.set_data(x1_traj[:k+1], x2_traj[:k+1])
        h_pt.set_data([x1_traj[k]], [x2_traj[k]])

        h_head.set_data(
            [x1_traj[k], x1_traj[k] + Lh*np.cos(psi_traj[k])],
            [x2_traj[k], x2_traj[k] + Lh*np.sin(psi_traj[k])]
        )

        plt.pause(0.05)

    plt.show()


if __name__ == "__main__":
    main()