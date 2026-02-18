"""
Task 3 – Compare UL1 from Set1 and Set2; detect clock offset via cross-correlation.
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
mat2 = scipy.io.loadmat('VoltageSet2.mat')

ts1  = parse_ts(mat1['ts']);  UL1_1 = mat1['UL1'].flatten()
ts2  = parse_ts(mat2['ts']);  UL1_2 = mat2['UL1'].flatten()

df1 = pd.Series(UL1_1, index=ts1)
df2 = pd.Series(UL1_2, index=ts2)

offset_at_start = (ts1[0] - ts2[0]).total_seconds()
print(f"Timestamp offset at start of recordings: {offset_at_start:.3f} s")

# ── Plot 1: full trace comparison + zoom ─────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

ax = axes[0]
ax.plot(ts1, UL1_1, lw=0.8, color='steelblue',  label='Set 1 UL1', alpha=0.9)
ax.plot(ts2, UL1_2, lw=0.8, color='darkorange', label='Set 2 UL1', alpha=0.8)
ax.set_title('Task 3 – UL1: Set 1 vs Set 2 (full 3-hour window)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage [V]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# zoom into first 10 minutes so any clock shift is visible
t_end = ts1[0] + pd.Timedelta(minutes=10)
m1 = ts1 <= t_end
m2 = ts2 <= t_end
ax2 = axes[1]
ax2.plot(ts1[m1], UL1_1[m1], 'o-', ms=3, lw=1, color='steelblue',  label='Set 1 UL1')
ax2.plot(ts2[m2], UL1_2[m2], 's-', ms=3, lw=1, color='darkorange', label='Set 2 UL1')
ax2.set_title(f'Zoom: First 10 min  (start offset ≈ {offset_at_start:.2f} s)',
              fontsize=11)
ax2.set_ylabel('Voltage [V]')
ax2.set_xlabel('Time')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_comparison.png', dpi=150)
plt.show()

# ── Cross-correlation to validate suspected offset ────────────────────────────
# Resample both series to a uniform 5-second grid
t_start = max(ts1[0], ts2[0])
t_stop  = min(ts1[-1], ts2[-1])
grid = pd.date_range(t_start, t_stop, freq='5s')

s1 = df1.reindex(grid, method='nearest', tolerance=pd.Timedelta('3s')).ffill().bfill()
s2 = df2.reindex(grid, method='nearest', tolerance=pd.Timedelta('3s')).ffill().bfill()

# Zero-mean signals for cross-correlation
x1 = s1.values - s1.mean()
x2 = s2.values - s2.mean()

# Full cross-correlation
xcorr = np.correlate(x1, x2, mode='full')
mid = len(xcorr) // 2

# Inspect ±600 seconds (±120 samples at 5 s resolution)
max_lag_samples = 120
lags_s = np.arange(-max_lag_samples, max_lag_samples + 1) * 5
window = xcorr[mid - max_lag_samples : mid + max_lag_samples + 1]
xcorr_norm = window / np.max(np.abs(window))

peak_idx   = np.argmax(np.abs(xcorr_norm))
peak_lag_s = lags_s[peak_idx]
print(f"Cross-correlation peak at lag = {peak_lag_s} s  "
      f"(|r| = {np.abs(xcorr_norm[peak_idx]):.4f})")
print(f"Estimated clock offset: Set2 leads Set1 by {-peak_lag_s} s")

# ── Plot 2: cross-correlation ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(lags_s, np.abs(xcorr_norm), color='purple', lw=1.2)
ax.axvline(peak_lag_s, color='red', ls='--', lw=1.5,
           label=f'Peak at lag = {peak_lag_s} s')
ax.set_title('Task 3 – Cross-correlation |r(k)| between Set 1 and Set 2 UL1',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Lag [s]')
ax.set_ylabel('|r(k)|  (normalised)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('task3_crosscorr.png', dpi=150)
plt.show()