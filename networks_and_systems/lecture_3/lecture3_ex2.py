"""
Task 2 – 15-minute Aggregation with Different Time Offsets (0s, 5s, 30s, 60s)
         and RMSE of shifted aggregations vs. the 0s reference.
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

# ── 15-min aggregation for each offset ───────────────────────────────────────
offsets_s = [0, 5, 30, 60]
agg_results = {}

for off in offsets_s:
    # Shift timestamps backward by 'off' seconds so that resample cuts
    # fall at the desired boundary position, then shift the result back.
    shifted_index = ts1 - pd.Timedelta(seconds=off)
    s = pd.Series(UL1, index=shifted_index)
    agg = s.resample('15min').mean()
    agg_results[off] = agg   # index is still on the shifted grid

# ── RMSE vs. 0s reference ─────────────────────────────────────────────────────
ref = agg_results[0]
rmse_vals = [0.0]    # RMSE of reference against itself is 0

for off in offsets_s[1:]:
    a = agg_results[off]
    n = min(len(ref), len(a))
    diff = ref.values[:n] - a.values[:n]
    valid = ~np.isnan(diff)
    rmse = np.sqrt(np.mean(diff[valid] ** 2))
    rmse_vals.append(rmse)
    print(f"RMSE (offset {off:2d}s vs 0s): {rmse:.6f} V")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = ['black', 'steelblue', 'darkorange', 'green']
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# --- aggregated traces ---
ax = axes[0]
for off, col in zip(offsets_s, colors):
    agg = agg_results[off]
    ax.step(agg.index, agg.values, where='post',
            lw=2.0 if off == 0 else 1.0,
            color=col, alpha=1.0 if off == 0 else 0.75,
            label=f'Offset {off}s')
ax.set_title('Task 2 – 15-min Aggregated Mean (different start offsets)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage [V]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- RMSE bar chart ---
ax2 = axes[1]
ax2.bar([f'{o}s' for o in offsets_s], rmse_vals, color=colors, alpha=0.75)
ax2.set_title('RMSE of Shifted 15-min Aggregation vs. 0s Offset')
ax2.set_xlabel('Offset')
ax2.set_ylabel('RMSE [V]')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(rmse_vals):
    ax2.text(i, v + 0.001, f'{v:.5f} V', ha='center', fontsize=10)

plt.tight_layout()
#plt.savefig('task2_aggregation.png', dpi=150)
plt.show()