# EKF and KF Exercise Solution (Process + Explanation)

## Overview

We consider the nonlinear scalar state-space system

\[
x_k = a\sin(x_{k-1}+\phi_f)+b u_{k-1}+w_{k-1}
\]
\[
y_k = \sin(c x_k+\phi_h)+v_k
\]

with Gaussian noise terms

\[
w_k \sim \mathcal N(0,Q), \qquad v_k \sim \mathcal N(0,R)
\]

and initial state

\[
x_0 \sim \mathcal N(\hat x_0, P_0).
\]

The goal is to:
1. Derive the Extended Kalman Filter (EKF) equations.
2. Simulate the nonlinear system.
3. Implement and compare a **linear KF** and an **EKF**.
4. Study performance under four parameter cases.

---

## Exercise 1 — Derive the EKF for the nonlinear system

### 1.1 Nonlinear model functions

We define the state transition and measurement functions as:

\[
f(x_{k-1},u_{k-1}) = a\sin(x_{k-1}+\phi_f)+b u_{k-1}
\]

\[
h(x_k)=\sin(c x_k+\phi_h)
\]

### 1.2 Jacobians (scalar case)

Because the EKF linearizes around the current estimate, we need the Jacobians:

- State Jacobian:
\[
F_k = \frac{\partial f}{\partial x}\bigg|_{\hat x_{k-1}^+}
= a\cos(\hat x_{k-1}^+ + \phi_f)
\]

- Measurement Jacobian:
\[
H_k = \frac{\partial h}{\partial x}\bigg|_{\hat x_k^-}
= c\cos(c\hat x_k^- + \phi_h)
\]

### 1.3 EKF recursion

Let \(\hat x_k^-\) and \(P_k^-\) be the predicted state and covariance, and \(\hat x_k^+\) and \(P_k^+\) be the updated state and covariance.

#### Initialization
\[
\hat x_0^+ = \hat x_0,\qquad P_0^+=P_0
\]

#### Time update (prediction)
\[
\hat x_k^- = f(\hat x_{k-1}^+,u_{k-1})
= a\sin(\hat x_{k-1}^+ + \phi_f)+b u_{k-1}
\]

\[
P_k^- = F_k P_{k-1}^+ F_k^T + Q
\]

(Scalar case: \(P_k^- = F_k^2P_{k-1}^+ + Q\))

#### Measurement update (correction)
\[
\hat y_k^- = h(\hat x_k^-)=\sin(c\hat x_k^- + \phi_h)
\]

\[
S_k = H_k P_k^- H_k^T + R
\]

\[
K_k = P_k^- H_k^T S_k^{-1}
\]

\[
\hat x_k^+ = \hat x_k^- + K_k (y_k - \hat y_k^-)
\]

\[
P_k^+ = (I-K_kH_k)P_k^-(I-K_kH_k)^T + K_k R K_k^T
\]

The covariance update above is the **Joseph form**, which is numerically more stable.

---

## Exercise 2 — Simulate the nonlinear system

### 2.1 Purpose of the simulation
The simulation generates:
- a known input sequence \(u_k\),
- the true hidden state \(x_k\),
- noisy nonlinear measurements \(y_k\).

This gives a dataset for evaluating both filters on the **same realization**.

### 2.2 Simulation steps
For each time step:
1. Generate process noise \(w_k\sim\mathcal N(0,Q)\).
2. Propagate the state using the nonlinear dynamics.
3. Generate measurement noise \(v_k\sim\mathcal N(0,R)\).
4. Compute the nonlinear measurement.

### 2.3 Important comparison rule
To compare KF and EKF fairly:
- simulate **once**,
- then run **both filters** on the same \(u_k\) and \(y_k\).

If the system is re-simulated between filters, the noise changes and the comparison is no longer fair.

---

## Exercise 3 — Implement the linear KF and the EKF

### 3.1 Linear KF model (approximate)
The linear KF uses a small-signal approximation around \(x \approx 0\):

\[
\sin(x)\approx x
\]

which leads to the nominal linear model:

\[
x_k \approx a x_{k-1} + b u_{k-1} + w_{k-1}
\]
\[
y_k \approx c x_k + v_k
\]

This is only a good model when:
- the state remains small,
- \(\phi_f=0\),
- \(\phi_h=0\),
- and the measurement nonlinearity is not too strong.

### 3.2 EKF model
The EKF uses the full nonlinear functions \(f(\cdot)\) and \(h(\cdot)\) and updates the linearization at every time step using the Jacobians \(F_k\) and \(H_k\).

### 3.3 What is compared
For each filter we evaluate:
- **State estimation accuracy** (RMSE between true state and estimated state)
- Optional: innovation sequences (to inspect model mismatch qualitatively)

---

## Exercise 4 — Compare performance for the four parameter cases

We study the following cases:

### (a) Initial parameters
\[
\phi_f = 0,\quad \phi_h = 0,\quad c = 1
\]

**Expected behavior:**
- Nonlinearity is moderate.
- Linear KF may work reasonably if the state stays in a small range.
- EKF should still perform better because it matches the nonlinear model.

---

### (b) Measurement phase shift: \(\phi_h = \pi/16\)
\[
y_k = \sin(c x_k + \phi_h)+v_k
\]

**Why KF degrades:**
The linear KF assumes \(y_k \approx c x_k + v_k\), but the phase shift changes both:
- the effective offset,
- and the local slope of the measurement function.

This creates a systematic measurement-model mismatch.

**Why EKF helps:**
The EKF uses:
\[
h(x)=\sin(c x + \phi_h)
\]
and the correct Jacobian
\[
H_k = c\cos(c\hat x_k^-+\phi_h)
\]
so it adapts to the phase shift.

---

### (c) State phase shift: \(\phi_f = \pi/16\)
\[
x_k = a\sin(x_{k-1}+\phi_f)+b u_{k-1}+w_{k-1}
\]

**Why KF degrades more strongly:**
This is a mismatch in the **dynamics model**, not only in the measurement.  
Prediction errors accumulate over time, so the effect is often more severe than case (b).

**Why EKF helps:**
The EKF predicts with the correct nonlinear state equation and linearizes locally using:
\[
F_k = a\cos(\hat x_{k-1}^+ + \phi_f)
\]

---

### (d) Strongly nonlinear measurement: \(c = 10\)
\[
y_k = \sin(10 x_k + \phi_h)+v_k
\]

**Why KF performs poorly:**
The measurement is now:
- highly nonlinear,
- periodic,
- bounded in \([-1,1]\),
- with rapidly varying local sensitivity.

A fixed linear measurement model is generally a poor approximation.

**Why EKF performs better:**
The EKF updates the measurement slope through:
\[
H_k = 10\cos(10\hat x_k^-+\phi_h)
\]
which allows it to follow local changes in sensitivity.

---

## Performance metric and interpretation

### RMSE
The main metric is the posterior state RMSE:
\[
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{k=1}^N (x_k - \hat x_k^+)^2}
\]

A lower RMSE means better tracking of the true hidden state.

### Expected trend
- EKF should outperform KF in all cases.
- The performance gap should increase as nonlinearity/model mismatch increases:
  - small in case (a),
  - larger in (b),
  - even larger in (c),
  - often largest in (d).

---

## How the provided MATLAB script is organized

The clean script performs the following steps automatically:

1. **Set base parameters** (\(a,b,Q,R,P_0\), etc.)
2. **Define the four exercise cases**
3. For each case:
   - Simulate the nonlinear system
   - Run the linear KF
   - Run the EKF
   - Compute RMSE
   - (Optional) repeat in Monte Carlo runs
4. Print a summary table of KF vs EKF RMSE
5. Plot one example trajectory (true state, KF estimate, EKF estimate, innovations)

---

## Final conclusion (ready to reuse)

The EKF consistently outperforms the linear KF because it accounts for the nonlinear state transition and measurement functions. The linear KF can still perform acceptably in the nominal case when the state remains near the operating point where \(\sin(x)\approx x\), but it degrades significantly when phase shifts are introduced or when the measurement nonlinearity becomes strong (e.g., \(c=10\)). The strongest degradation occurs when the mismatch is in the state dynamics, since prediction errors propagate recursively through time.