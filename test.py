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
npts = 200
dt = 0.05
dtassim = dt
ntimes = 200
oberrstdev = 1.0
oberrvar = oberrstdev**2
F = 8
diff_min = 0.5
diff_max = 2.5
deltaF = 1.0 / 8.0
Fcorr = np.exp(-1) ** (1.0 / 3.0)
inflation = 1.05  # Covariance inflation factor

# -----------------------------
# Observation settings
# -----------------------------
# Set the fraction of state variables that are directly observed.
# 1.0 = full observation, 0.5 = observe every other variable (half),
# 0.25 = observe every 4th variable (quarter), etc.
obs_fraction = 0.5  # <-- change to 0.25 for quarter observation
rs_truth = np.random.RandomState(42)
rs_ens = np.random.RandomState(123)

# Observation operator and localization
# Observe a subset of variables determined by obs_fraction.
step = int(1 / obs_fraction)
obs_indices = np.arange(0, npts, step)
h = np.eye(npts)[obs_indices]      # shape: (n_obs, npts)
covlocal = np.ones((npts, npts))   # keep full covariance localization for now

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
def run_filter(method_fn, dropout=False, rate=0.1, members=None):
    rs = np.random.RandomState(999)
    ne = members if members is not None else nens
    model = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, members=ne, dt=dt,
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
# Time series diagnostics: RMSE, spread, truth vs mean
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Run both filters
nens = 40  # or any value you'd like to compare
rmse_bulk, spread_bulk = run_filter(bulk_enkf, members=nens)
rmse_drop, spread_drop = run_filter(dropout_enkf, dropout=True, rate=0.10, members=nens)

axs[0].plot(rmse_bulk, label='Bulk EnKF')
axs[0].plot(rmse_drop, label='Dropout EnKF (p=0.1)')
axs[0].set_ylabel("RMSE")
axs[0].legend()
axs[0].set_title("RMSE over time")

axs[1].plot(spread_bulk, label='Bulk EnKF')
axs[1].plot(spread_drop, label='Dropout EnKF (p=0.1)')
axs[1].set_ylabel("Ensemble spread")
axs[1].legend()
axs[1].set_title("Spread over time")

# -----------------------------
# Truth vs mean (all variables as 2D image)
# -----------------------------
model_truth = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, dt=dt,
                  diff_max=diff_max, diff_min=diff_min, rs=np.random.RandomState(42))
for _ in range(100): model_truth.advance()
truth_traj_all = []
for _ in range(ntimes):
    model_truth.advance()
    truth_traj_all.append(model_truth.x[0])
truth_traj_all = np.array(truth_traj_all)

model_bulk = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, members=nens, dt=dt,
                 diff_max=diff_max, diff_min=diff_min, rs=np.random.RandomState(999))
mean_traj_bulk_all = []
for t in range(ntimes):
    xmean = model_bulk.x.mean(axis=0)
    xprime = model_bulk.x - xmean
    xmean, xprime = bulk_enkf(xmean, xprime, h, obs[t], oberrvar, covlocal, np.random)
    model_bulk.x = xmean + xprime
    model_bulk.advance()
    mean_traj_bulk_all.append(xmean)
mean_traj_bulk_all = np.array(mean_traj_bulk_all)

model_drop = L96(n=npts, F=F, deltaF=deltaF, Fcorr=Fcorr, members=nens, dt=dt,
                 diff_max=diff_max, diff_min=diff_min, rs=np.random.RandomState(888))
mean_traj_drop_all = []
for t in range(ntimes):
    xmean = model_drop.x.mean(axis=0)
    xprime = model_drop.x - xmean
    xmean, xprime = dropout_enkf(xmean, xprime, h, obs[t], oberrvar, covlocal, np.random, dropout_rate=0.10)
    model_drop.x = xmean + xprime
    model_drop.advance()
    mean_traj_drop_all.append(xmean)
mean_traj_drop_all = np.array(mean_traj_drop_all)

fig_all, axs_all = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
im0 = axs_all[0].imshow(truth_traj_all.T, aspect='auto', origin='lower', cmap='viridis')
axs_all[0].set_ylabel("State index")
axs_all[0].set_title("Truth trajectory")
plt.colorbar(im0, ax=axs_all[0])

im1 = axs_all[1].imshow(mean_traj_bulk_all.T, aspect='auto', origin='lower', cmap='viridis')
axs_all[1].set_ylabel("State index")
axs_all[1].set_title("Bulk EnKF ensemble mean")

im2 = axs_all[2].imshow(mean_traj_drop_all.T, aspect='auto', origin='lower', cmap='viridis')
axs_all[2].set_ylabel("State index")
axs_all[2].set_xlabel("Time step")
axs_all[2].set_title("Dropout EnKF (p=0.1) ensemble mean")

plt.colorbar(im1, ax=axs_all[1])
plt.colorbar(im2, ax=axs_all[2])
plt.tight_layout()
plt.savefig("trajectory_vs_all.png")
plt.show()

err_bulk_img = np.abs(mean_traj_bulk_all - truth_traj_all)
err_drop_img = np.abs(mean_traj_drop_all - truth_traj_all)

fig_err, axs_err = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ie1 = axs_err[0].imshow(err_bulk_img.T, aspect='auto', origin='lower', cmap='inferno')
axs_err[0].set_ylabel("State index")
axs_err[0].set_title("Abs Error: Bulk EnKF - Truth")
plt.colorbar(ie1, ax=axs_err[0])

ie2 = axs_err[1].imshow(err_drop_img.T, aspect='auto', origin='lower', cmap='inferno')
axs_err[1].set_ylabel("State index")
axs_err[1].set_xlabel("Time step")
axs_err[1].set_title("Abs Error: Dropout EnKF (p=0.1) - Truth")
plt.colorbar(ie2, ax=axs_err[1])

plt.tight_layout()
plt.savefig("error_map_all.png")
plt.show()

# -----------------------------
# Parameter sweep: ensemble size vs. median analysis RMSE
# -----------------------------
ensemble_sizes = [20, 40, 80, 160]
methods = {
    'bulk_enkf': lambda ne: run_filter(bulk_enkf, members=ne)[0], 
    'dropout_0.10': lambda ne: run_filter(dropout_enkf, dropout=True, rate=0.10, members=ne)[0],
    'dropout_0.20': lambda ne: run_filter(dropout_enkf, dropout=True, rate=0.20, members=ne)[0],
}

median_rmse = {name: [] for name in methods}

for ne in ensemble_sizes:
    for name, fn in methods.items():
        # update ensemble size
        global nens
        nens = ne
        err = fn(ne)
        median_rmse[name].append(np.median(err))

# -----------------------------
# Plot RMSE vs. ensemble size
# -----------------------------
plt.figure(figsize=(8,5))
for name, rmses in median_rmse.items():
    plt.plot(ensemble_sizes, rmses, marker='o', label=name)
plt.xlabel('Ensemble size N_e')
plt.ylabel('Median Analysis RMSE')
plt.title('Ensemble size sweep')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("ensemble_size_sweep.png")
plt.show()

# # Exit after sweep
# import sys; sys.exit()
# %%
# -----------------------------
# 2-D heatmaps: (y, gamma) vs max abs-err
# -----------------------------
# y_vals = np.linspace(0.0, 2.0, 30)          # 观测值轴
# g_vals = np.linspace(0.25, 2.0, 30)         # 观测误差σ轴 (gamma)
# methods_panel = {
#     'bulk':          {'fn': bulk_enkf,    'drop': False, 'p': 0.0},
#     'dropout_0.10':  {'fn': dropout_enkf, 'drop': True,  'p': 0.10},
# }
# ens_sizes_panel = [20, 80]                 # 上、下两行

# def single_analysis(prior_mean, prior_var, y, gamma, method_spec, N_e):
#     """1D Gauss–Gauss：x~N(prior_mean,prior_var), y=x+ε, ε~N(0,γ²)"""
#     # 构造 1 维 ensemble
#     ens = prior_mean + np.sqrt(prior_var) * np.random.randn(N_e)
#     xmean = ens.mean()
#     xprime = ens - xmean
#     h1 = np.array([[1.0]])                # H=1
#     covloc = np.ones((1,1))
#     if method_spec['drop']:
#         xmean, xprime = method_spec['fn'](xmean, xprime[:,None], h1,
#                                           np.array([y]), gamma**2,
#                                           covloc, np.random, dropout_rate=method_spec['p'])
#     else:
#         xmean, xprime = method_spec['fn'](xmean, xprime[:,None], h1,
#                                           np.array([y]), gamma**2,
#                                           covloc, np.random)
#     # 返回分析均值
#     return xmean

# # 真值固定 0，先验均值 0，方差 1
# prior_mu, prior_var = 0.0, 1.0

# fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

# for row, N_e in enumerate([20, 80]):
#     for col, (mname, mspec) in enumerate(methods_panel.items()):
#         err_mat = np.zeros((len(g_vals), len(y_vals)))
#         for i_g, g in enumerate(g_vals):
#             for j_y, yy in enumerate(y_vals):
#                 errs = []
#                 for _ in range(50):
#                     xa = single_analysis(prior_mu, prior_var, yy, g, mspec, N_e)
#                     kalman_mu = (prior_var / (prior_var + g**2)) * yy
#                     errs.append(abs(xa - kalman_mu))
#                 err_mat[i_g,j_y] = np.median(errs)
#         im = axs[row, col].imshow(err_mat, origin='lower',
#                                   extent=[y_vals[0], y_vals[-1], g_vals[0], g_vals[-1]],
#                                   vmin=0, vmax=0.6, cmap='turbo', aspect='auto')
#         axs[row, col].set_title(f"{mname.upper()} Max Errors", fontsize=10)
#         axs[row, col].set_xlabel("$y$")
#         axs[row, col].set_ylabel("$\gamma$")
#         axs[row, col].text(0.95, 0.05, f"N = {N_e}", transform=axs[row, col].transAxes,
#                            ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax, label="Median |error|")
# fig.suptitle("Max abs error comparison: EnKF vs Dropout-EnKF", fontsize=12)
# plt.tight_layout(rect=[0, 0, 0.86, 0.95])
# plt.savefig("panel_heatmap_comparison.png")
# plt.show()