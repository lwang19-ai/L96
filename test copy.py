# %%
import numpy as np
import matplotlib.pyplot as plt
from L96 import L96
from enkf import bulk_enkf, dropout_enkf, serial_ensrf, bulk_ensrf
from enkf import etkf

np.seterr(all="raise")

# -----------------------------
# Configuration
# -----------------------------
nens = 20
npts = 40
dt = 0.05
dtassim = dt
ntimes = 500
oberrstdev = 1.0
oberrvar = oberrstdev**2
F = 8
diff_min = 0.5
diff_max = 2.5
deltaF = 1.0 / 8.0
Fcorr = np.exp(-1) ** (1.0 / 3.0)
inflation = 1.05  # Covariance inflation factor
rs_truth = np.random.RandomState(42)
rs_ens = np.random.RandomState(123)

# Observation operator and localization
# Full observation of all variables
h = np.eye(npts)
covlocal = np.ones((npts, npts))

# -----------------------------
# Prepare truth and observations
# -----------------------------
truth = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, dt=dt,
            diff_max=diff_max, diff_min=diff_min, rs=rs_truth)

for _ in range(100):
    truth.advance()

obs = []
xtruth = []
for _ in range(ntimes):
    truth.advance()
    obs.append(h @ truth.x[0] + oberrstdev * rs_ens.randn(h.shape[0]))
    xtruth.append(truth.x[0].copy())
obs = np.array(obs)
xtruth = np.array(xtruth)

# -----------------------------
# Filter runners
# -----------------------------
def run_filter(method_fn, dropout=False, rate=0.1):
    rs = np.random.RandomState(999)
    model = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, members=nens, dt=dt,
                diff_max=diff_max, diff_min=diff_min, rs=rs)

    analerrs, analsprds = [], []
    for t in range(ntimes):
        xmean = model.x.mean(axis=0)
        xprime = model.x - xmean

        if method_fn is etkf:
            xmean, xprime = method_fn(xmean, xprime, h, obs[t], oberrvar)
        elif dropout:
            xmean, xprime = method_fn(xmean, xprime, h, obs[t], oberrvar, covlocal, rs, dropout_rate=rate)
        else:
            xmean, xprime = method_fn(xmean, xprime, h, obs[t], oberrvar, covlocal, rs)

        # Apply multiplicative inflation to anomalies
        xprime = xprime * np.sqrt(inflation)

        analerrs.append(np.sqrt(np.mean((xmean - xtruth[t])**2)))
        analsprds.append(np.sqrt(np.mean(np.var(xprime, axis=0))))

        model.x = xmean + xprime
        model.x = np.clip(model.x, -1e3, 1e3)
        model.advance()

    return np.array(analerrs), np.array(analsprds)

# -----------------------------
# Run experiments
# -----------------------------
results = {}

# Baseline methods
# results['serial_ensrf'] = run_filter(serial_ensrf)
# results['bulk_ensrf'] = run_filter(bulk_ensrf)
# results['etkf'] = run_filter(etkf)
results['bulk_enkf'] = run_filter(bulk_enkf)

# Dropout variants
dropout_rates = [ 0.1, 0.2]
for rate in dropout_rates:
    label = f"dropout_enkf_{rate:.2f}"
    results[label] = run_filter(dropout_enkf, dropout=True, rate=rate)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for label, (err, _) in results.items():
    plt.plot(err, label=f"{label} (RMSE)")
plt.ylabel("Analysis RMSE")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
for label, (_, spr) in results.items():
    plt.plot(spr, label=f"{label} (Spread)")
plt.ylabel("Analysis Spread")
plt.xlabel("Assimilation Step")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("enkf_multiple_comparison.png")
plt.show()