"""
Task 4 – Markov Modulated Process: derive Q (generator) and E (emission) matrices.
         Uses N=10 equidistant voltage states covering the range of UL1 (Trace 1).

Approach
--------
1. Quantise voltage trace into N discrete states (equidistant boundaries).
2. Count state-to-state transitions to form the embedded-chain count matrix.
3. Row-normalise to get the transition probability matrix P.
4. Estimate the holding-time rate λ_i = 1 / mean_hold_i for each state.
5. Build the generator matrix  Q[i,j] = λ_i * P[i,j]  (i≠j),  Q[i,i] = −λ_i.
6. Define the emission matrix  E = diag(state-centre voltages).
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Helper: parse MATLAB timestamp cell array ─────────────────────────────────
def parse_ts(ts_array):
    flat = ts_array.flatten()
    times = []
    for item in flat:
        s = item.flat[0] if hasattr(item, 'flat') else item
        if hasattr(s, 'flat'):
            s = s.flat[0]
        times.append(pd.Timestamp(str(s)))
    return pd.DatetimeIndex(times)


# ── Load data ─────────────────────────────────────────────────────────────────
mat1 = scipy.io.loadmat('VoltageSet1.mat')
ts1  = parse_ts(mat1['ts'])
UL1  = mat1['UL1'].flatten()

# ── State space: N equidistant voltage bins ───────────────────────────────────
N     = 10
v_min, v_max = UL1.min(), UL1.max()
edges   = np.linspace(v_min, v_max, N + 1)          # N+1 boundary values
centers = 0.5 * (edges[:-1] + edges[1:])            # N state-centre voltages

print(f"Voltage range  : [{v_min:.2f}, {v_max:.2f}] V")
print(f"State boundaries:\n  {np.round(edges, 3)}")
print(f"State centres   :\n  {np.round(centers, 3)}")

# Assign each sample to a state index 0 … N-1
states = np.digitize(UL1, edges[1:-1])   # interior edges → 0-indexed states

# ── Step 1: Embedded Markov chain – count matrix ──────────────────────────────
P_count = np.zeros((N, N), dtype=float)
for t in range(len(states) - 1):
    i, j = states[t], states[t + 1]
    P_count[i, j] += 1

# ── Step 2: Row-normalise to get transition probability matrix P ──────────────
row_sums = P_count.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1          # avoid division by zero for empty states
P = P_count / row_sums

# ── Step 3: Mean holding time per state ──────────────────────────────────────
dt_arr = np.array([(ts1[t+1] - ts1[t]).total_seconds()
                   for t in range(len(states) - 1)])

mean_hold = np.zeros(N)
for i in range(N):
    mask_i = (states[:-1] == i)
    mean_hold[i] = dt_arr[mask_i].mean() if mask_i.sum() > 0 else dt_arr.mean()

lambda_i = 1.0 / mean_hold          # transition rate out of each state

# ── Step 4: Generator matrix Q ───────────────────────────────────────────────
#   Q[i,j] = λ_i * P[i,j]   for i ≠ j
#   Q[i,i] = −λ_i            (rows sum to 0)
Q = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            Q[i, j] = lambda_i[i] * P[i, j]
    Q[i, i] = -np.sum(Q[i, np.arange(N) != i])   # enforce exact row-sum = 0

# ── Step 5: Emission matrix E (diagonal, state centre voltages) ───────────────
E = np.diag(centers)

# ── Print results ─────────────────────────────────────────────────────────────
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("\nP matrix (embedded Markov chain):")
print(P)
print("\nQ matrix (generator) [s⁻¹]:")
print(Q)
print("\nE matrix (emission / state voltages) [V]:")
print(E)
print("\nRow sums of Q (should be ~0):")
print(np.round(Q.sum(axis=1), 10))

# ── Stationary distribution π from left null-vector of Q ─────────────────────
evals, evecs = np.linalg.eig(Q.T)
stat_idx = np.argmin(np.abs(evals))
pi = np.real(evecs[:, stat_idx])
pi = np.abs(pi) / np.abs(pi).sum()

emp_freq = np.array([np.sum(states == i) / len(states) for i in range(N)])
print("\nStationary distribution π:")
print(np.round(pi, 4))
print("\nEmpirical state frequencies:")
print(np.round(emp_freq, 4))

# ── Plots ─────────────────────────────────────────────────────────────────────
tick_labels = [f'{c:.1f}' for c in centers]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- state sequence ---
ax = axes[0, 0]
ax.plot(ts1, states, lw=0.6, color='steelblue', alpha=0.8)
ax.set_title('State Sequence (N=10 states)')
ax.set_ylabel('State index')
ax.set_xlabel('Time')
ax.grid(True, alpha=0.3)

# --- P matrix heatmap ---
ax = axes[0, 1]
im = ax.imshow(P, cmap='Blues', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax)
ax.set_title('P Matrix – Embedded Markov Chain')
ax.set_xlabel('To state j')
ax.set_ylabel('From state i')
ax.set_xticks(range(N)); ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
ax.set_yticks(range(N)); ax.set_yticklabels(tick_labels, fontsize=7)

# --- Q matrix heatmap (off-diagonal) ---
Q_plot = Q.copy()
np.fill_diagonal(Q_plot, np.nan)
ax = axes[1, 0]
im2 = ax.imshow(Q_plot, cmap='Reds', aspect='auto')
plt.colorbar(im2, ax=ax)
ax.set_title('Q Matrix – Generator (off-diagonal rates [s⁻¹])')
ax.set_xlabel('To state j')
ax.set_ylabel('From state i')
ax.set_xticks(range(N)); ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
ax.set_yticks(range(N)); ax.set_yticklabels(tick_labels, fontsize=7)

# --- stationary π vs empirical ---
ax = axes[1, 1]
x_pos = np.arange(N)
w = 0.35
ax.bar(x_pos - w/2, pi,       w, label='Stationary π (Q)', color='steelblue',  alpha=0.8)
ax.bar(x_pos + w/2, emp_freq, w, label='Empirical freq',   color='darkorange', alpha=0.8)
ax.set_title('Stationary π vs. Empirical State Frequencies')
ax.set_xlabel('State (centre voltage)')
ax.set_ylabel('Probability')
ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Task 4 – Markov Modulated Process (N=10)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
#plt.savefig('task4_markov.png', dpi=150, bbox_inches='tight')
plt.show()