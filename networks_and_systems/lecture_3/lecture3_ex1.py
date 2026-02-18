"""
Task 1 – UL1 Trace: Mean, Standard Deviation, RMS, and Plot
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

# ── Statistics ────────────────────────────────────────────────────────────────
mean_v = np.mean(UL1)
std_v  = np.std(UL1, ddof=1)
rms_v  = np.sqrt(np.mean(UL1 ** 2))

dt_s = np.array([(ts1[i+1] - ts1[i]).total_seconds() for i in range(len(ts1)-1)])

print(f"N samples  : {len(UL1)}")
print(f"Duration   : {ts1[0]}  →  {ts1[-1]}")
print(f"Sample Δt  : mean={dt_s.mean():.2f}s  std={dt_s.std():.3f}s  "
      f"min={dt_s.min():.2f}s  max={dt_s.max():.2f}s")
print(f"Mean       : {mean_v:.4f} V")
print(f"Std dev    : {std_v:.4f} V")
print(f"RMS        : {rms_v:.4f} V")
print(f"Min / Max  : {UL1.min():.2f} V  /  {UL1.max():.2f} V")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# --- voltage trace ---
ax = axes[0]
ax.plot(ts1, UL1, lw=0.8, color='steelblue')
ax.axhline(mean_v, color='red', ls='--', lw=1.3, label=f'Mean = {mean_v:.2f} V')
ax.fill_between(ts1, mean_v - std_v, mean_v + std_v,
                alpha=0.15, color='red', label=f'±1σ = {std_v:.2f} V')
ax.set_title('Task 1 – UL1 Trace (Set 1)', fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage [V]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- histogram ---
ax2 = axes[1]
ax2.hist(UL1, bins=60, color='steelblue', edgecolor='white', lw=0.3)
ax2.axvline(mean_v, color='red', ls='--', lw=1.5, label=f'Mean = {mean_v:.2f} V')
ax2.axvline(mean_v - std_v, color='orange', ls=':', lw=1.5)
ax2.axvline(mean_v + std_v, color='orange', ls=':', lw=1.5, label='±1σ')
ax2.set_title('Voltage Distribution')
ax2.set_xlabel('Voltage [V]')
ax2.set_ylabel('Count')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- inter-sample interval (data quality) ---
ax3 = axes[2]
ax3.plot(ts1[1:], dt_s, lw=0.7, color='darkorange', alpha=0.9)
ax3.axhline(dt_s.mean(), color='red', ls='--', lw=1.2,
            label=f'Mean Δt = {dt_s.mean():.2f} s')
ax3.set_title('Inter-sample Interval – Data Quality Check')
ax3.set_ylabel('Δt [s]')
ax3.set_xlabel('Time')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('task1_trace.png', dpi=150)
plt.show()