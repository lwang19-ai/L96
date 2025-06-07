
"""Ensemble square-root filters for the Lorenz 96 model."""
import sys

import numpy as np

from L96 import L96
from enkf import (
    serial_ensrf,
    bulk_ensrf,
    dropout_enkf,
    etkf,
    letkf,
    etkf_modens,
    serial_ensrf_modens,
    bulk_enkf,
    getkf,
    getkf_modens,
)

np.seterr(all="raise")  # raise error when overflow occurs

if len(sys.argv) < 3:
    msg = """
python L96ensrf.py covlocal method smooth_len gaussian

All variables are observed; assimilation interval given by dtassim,
nens ensemble members, observation error standard deviation = oberrstdev,
observation operator is smooth_len pt boxcar running mean or gaussian.

Time-mean error and spread stats are printed to standard output.

covlocal:  localization distance (distance at which Gaspari-Cohn
polynomial goes to zero).

method:  =0 for serial Potter method
         =1 for bulk Potter method (all obs at once)
         =2 for ETKF (no localization applied)
         =3 for LETKF (using observation localization)
         =4 for serial Potter method with localization via modulation
             ensemble
         =5 for ETKF with modulation ensemble
         =6 for ETKF with modulation ensemble and perturbed obs
         =7 for serial Potter method using sqrt of localized Pb ensemble
         =8 for bulk EnKF (all obs at once) with perturbed obs
         =9 for GETKF (no localization applied)
         =10 for GETKF (with modulated ensemble)
         =11 for ETKF with modulation ensemble and stochastic
              subsampling
         =12 for ETKF with modulation ensemble and 'adjusted' perturbed
              obs
         =13 for 'DEnKF' approx to ETKF with modulated ensemble
         =14 for 'DEnKF' approx to bulk Potter method.

         =15 for Dropout-EnKF (bulk EnKF with random dropout regularization)

covinflate1, covinflate2: (optional) inflation parameters corresponding
to a and b in Hodyss and Campbell. If not specified, a=b=1. If
covinflate2 <= 0, relaxation to prior spread (RTPS) inflation is used
with a relaxation coefficient equal to covinflate1.
"""
    raise SystemExit(msg)

corrl = float(sys.argv[1])
method = int(sys.argv[2])
smooth_len = int(sys.argv[3])
use_gaussian = bool(int(sys.argv[4]))

covinflate1 = 1.0
covinflate2 = 1.0
if len(sys.argv) > 5:
    covinflate1 = float(sys.argv[5])
    covinflate2 = float(sys.argv[6])

ntstart = 1000  # time steps to spin up truth run
ntimes = 11000  # observation times
nens = 8  # ensemble members
oberrstdev = 0.1
oberrvar = oberrstdev**2
verbose = False  # print error stats every time if True

# Model parameters for truth run
dt = 0.05
npts = 80
dtassim = dt  # assimilation interval
diffusion_truth_max = 2.5
diffusion_truth_min = 0.5

# Forecast model parameters (same as truth run for perfect model example)
diffusion_max = diffusion_truth_max
diffusion_min = diffusion_truth_min

rstruth = np.random.RandomState(42)  # fixed seed for truth run
rsens = np.random.RandomState()  # varying seed for obs noise and ensemble ICs

# Model instance for truth (nature) run
F = 8
deltaF = 1.0 / 8.0
Fcorr = np.exp(-1) ** (1.0 / 3.0)  # efolding over n timesteps, n=3

model = L96(
    n=npts,
    F=F,
    deltaF=deltaF,
    Fcorr=Fcorr,
    dt=dt,
    diff_max=diffusion_truth_max,
    diff_min=diffusion_truth_min,
    rs=rstruth,
)

# Model instance for forecast ensemble
ensemble = L96(
    n=npts,
    F=F,
    deltaF=deltaF,
    Fcorr=Fcorr,
    members=nens,
    dt=dt,
    diff_max=diffusion_truth_max,
    diff_min=diffusion_truth_min,
    rs=rsens,
)

for _ in range(ntstart):
    model.advance()

# Sample obs from truth; compute climatology for model
xx = []
tt = []
for nt in range(ntimes):
    model.advance()
    xx.append(model.x[0])  # single member
    tt.append(float(nt) * model.dt)

xtruth = np.array(xx, float)
timetruth = np.array(tt, float)
xtruth_mean = xtruth.mean()
xprime = xtruth - xtruth_mean
xvar = np.sum(xprime**2, axis=0) / (ntimes - 1)
xtruth_stdev = np.sqrt(xvar.mean())

if verbose:
    print("climo for truth run:")
    print("x mean =", xtruth_mean)
    print("x stdev =", xtruth_stdev)

# Forward operator (identity obs); smoothing in forward operator
ndim = ensemble.n
h = np.eye(ndim)

if smooth_len > 0:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i - j)
            if i - j < -(ndim // 2):
                rr = float(ndim - j + i)
            if i - j > (ndim // 2):
                rr = float(i - ndim - j)
            r = abs(rr) / smooth_len
            if use_gaussian:
                h[j, i] = np.exp(-r**2)
            else:
                if r <= 1.0:
                    h[j, i] = 1.0
                else:
                    h[j, i] = 0.0
        h[j, :] = h[j, :] / h[j, :].sum()

obs = np.empty(xtruth.shape, xtruth.dtype)
for nt in range(xtruth.shape[0]):
    obs[nt] = np.dot(h, xtruth[nt])

obs = obs + oberrstdev * rsens.standard_normal(size=obs.shape)

# Spin up ensemble
ntot = xtruth.shape[0]
nspinup = ntstart
for _ in range(ntstart):
    ensemble.advance()

nsteps = int(dtassim / model.dt)
if verbose:
    print("ntstart, nspinup, ntot, nsteps =", ntstart, nspinup, ntot, nsteps)
if nsteps < 1:
    raise ValueError("assimilation interval must be at least one model time step")

def ensrf(ensemble, xmean, xprime, hmat, obs_val, oberrvar, covlocal, method=1, z=None):
    """Wrapper to call appropriate filter based on method flag."""
    if method == 0:
        return serial_ensrf(xmean, xprime, hmat, obs_val, oberrvar, covlocal, covlocal)
    if method == 1:
        return bulk_ensrf(xmean, xprime, hmat, obs_val, oberrvar, covlocal)
    if method == 2:
        return etkf(xmean, xprime, hmat, obs_val, oberrvar)
    if method == 3:
        return letkf(xmean, xprime, hmat, obs_val, oberrvar, covlocal)
    if method == 4:
        return serial_ensrf_modens(xmean, xprime, hmat, obs_val, oberrvar, covlocal, z)
    if method == 5:
        return etkf_modens(xmean, xprime, hmat, obs_val, oberrvar, covlocal, z)
    if method == 6:
        return etkf_modens(
            xmean, xprime, hmat, obs_val, oberrvar, covlocal, z, rs=rsens, po=True
        )
    if method == 7:
        return serial_ensrf_modens(xmean, xprime, hmat, obs_val, oberrvar, covlocal, None)
    if method == 8:
        return bulk_enkf(xmean, xprime, hmat, obs_val, oberrvar, covlocal, rsens)
    if method == 9:
        return getkf(xmean, xprime, hmat, obs_val, oberrvar)
    if method == 10:
        return getkf_modens(xmean, xprime, hmat, obs_val, oberrvar, covlocal, z)
    if method == 11:
        return etkf_modens(
            xmean, xprime, hmat, obs_val, oberrvar, covlocal, z, rs=rsens, ss=True
        )
    if method == 12:
        return etkf_modens(
            xmean,
            xprime,
            hmat,
            obs_val,
            oberrvar,
            covlocal,
            z,
            rs=rsens,
            po=True,
            adjust_obnoise=True,
        )
    if method == 13:
        return etkf_modens(xmean, xprime, hmat, obs_val, oberrvar, covlocal, z, denkf=True)
    if method == 14:
        return bulk_ensrf(xmean, xprime, hmat, obs_val, oberrvar, covlocal, denkf=True)
    if method == 15:
        return dropout_enkf(xmean, xprime, hmat, obs_val, oberrvar, covlocal, rsens, dropout_rate=0.1)
    raise ValueError("illegal value for enkf method flag")

# Define localization matrix
covlocal = np.eye(ndim)
xdep = model.diff_min + (model.diff_max - model.diff_min) * model.blend

if corrl < 2 * ndim:
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i - j)
            if i - j < -(ndim // 2):
                rr = float(ndim - j + i)
            if i - j > (ndim // 2):
                rr = float(i - ndim - j)
            r = abs(rr) / (xdep[i] * corrl)
            taper = 0.0
            rr2 = 2.0 * r
            if r <= 0.5:
                taper = (((-0.25 * rr2 + 0.5) * rr2 + 0.625) * rr2 - 5.0 / 3.0) * rr2**2 + 1.0
            elif 0.5 < r < 1.0:
                taper = (
                    ((((rr2 / 12.0 - 0.5) * rr2 + 0.625) * rr2 + 5.0 / 3.0) * rr2
                      - 5.0)
                    * rr2
                    + 4.0
                    - 2.0 / (3.0 * rr2)
                )
            covlocal[j, i] = taper
    covlocal = 0.5 * (covlocal + covlocal.T)

# Compute square root of covlocal if needed
if method in [4, 5, 6, 10, 11, 12, 13]:
    evals, eigs = np.linalg.eigh(covlocal)
    evals = np.where(evals > 1e-10, evals, 1e-10)
    evalsum = evals.sum()
    neig = 0
    frac = 0.0
    thresh = 0.99
    while frac < thresh:
        frac = evals[ndim - neig - 1 : ndim].sum() / evalsum
        neig += 1
    zz = (eigs * np.sqrt(evals / frac)).T
    z = zz[ndim - neig : ndim, :]
else:
    neig = 0
    z = None

# Run assimilation
fcsterr = []
fcsterr1 = []
fcstsprd = []
analerr = []
analsprd = []
diverged = False

fsprdmean = np.zeros(ndim, float)
fsprdobmean = np.zeros(ndim, float)
asprdmean = np.zeros(ndim, float)
ferrmean = np.zeros(ndim, float)
aerrmean = np.zeros(ndim, float)
corrmean = np.zeros(ndim, float)
corrhmean = np.zeros(ndim, float)

for nassim in range(0, ntot, nsteps):
    xmean = ensemble.x.mean(axis=0)
    xmean_b = xmean.copy()
    xprime = ensemble.x - xmean

    ferr = (xmean - xtruth[nassim]) ** 2
    if np.isnan(ferr.mean()):
        diverged = True
        break

    fsprd = np.sum(xprime**2, axis=0) / (ensemble.members - 1)
    corr = (xprime.T * xprime[:, ndim // 2]).sum(axis=1) / float(
        ensemble.members - 1
    )
    hxprime = np.dot(xprime, h)
    fsprdob = np.sum(hxprime**2, axis=0) / (ensemble.members - 1)
    corrh = (xprime.T * hxprime[:, ndim // 2]).sum(axis=1) / float(
        ensemble.members - 1
    )

    if nassim >= nspinup:
        fsprdmean += fsprd
        fsprdobmean += fsprdob
        corrmean += corr
        corrhmean += corrh
        ferrmean += ferr
        fcsterr.append(ferr.mean())
        fcstsprd.append(fsprd.mean())
        fcsterr1.append(xmean - xtruth[nassim])

    xmean, xprime = ensrf(
        ensemble, xmean, xprime, h, obs[nassim], oberrvar, covlocal, method=method, z=z
    )

    aerr = (xmean - xtruth[nassim]) ** 2
    asprd = np.sum(xprime**2, axis=0) / (ensemble.members - 1)

    if nassim >= nspinup:
        asprdmean += asprd
        aerrmean += aerr
        analerr.append(aerr.mean())
        analsprd.append(asprd.mean())

    if verbose:
        print(
            nassim,
            timetruth[nassim],
            np.sqrt(ferr.mean()),
            np.sqrt(fsprd.mean()),
            np.sqrt(aerr.mean()),
            np.sqrt(asprd.mean()),
        )

    if covinflate2 > 0:
        inc = xmean - xmean_b
        inf_fact = np.sqrt(
            covinflate1
            + (asprd / fsprd**2)
            * ((fsprd / ensemble.members)
               + covinflate2 * (2.0 * inc**2 / (ensemble.members - 1)))
        )
    else:
        asprd_sqrt = np.sqrt(asprd)
        fsprd_sqrt = np.sqrt(fsprd)
        inf_fact = 1.0 + covinflate1 * (fsprd_sqrt - asprd_sqrt) / asprd_sqrt

    xprime *= inf_fact

    # Run forecast model
    ensemble.x = xmean + xprime
    for _ in range(nsteps):
        ensemble.advance()

# Print out time-mean stats normalized by observation error
if diverged:
    print(
        method,
        len(fcsterr),
        corrl,
        covinflate1,
        covinflate2,
        oberrstdev,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        neig,
    )
else:
    ncount = len(fcstsprd)
    fcsterr = np.array(fcsterr)
    fcsterr1 = np.array(fcsterr1)
    fcstsprd = np.array(fcstsprd)
    analerr = np.array(analerr)
    analsprd = np.array(analsprd)

    fstdev = np.sqrt(fcstsprd.mean())
    astdev = np.sqrt(analsprd.mean())

    asprdmean /= ncount
    aerrmean /= ncount
    fsprdmean /= ncount
    fsprdobmean /= ncount
    corrmean /= ncount
    corrhmean /= ncount

    fstd = np.sqrt(fsprdmean)
    fstdob = np.sqrt(fsprdobmean)

    covmean = corrmean
    corrmean = corrmean / (fstd * fstd[ndim // 2])
    covhmean = corrhmean
    corrhmean = corrhmean / (fstd * fstdob[ndim // 2])

    fcsterrcorr = (
        fcsterr1.T * fcsterr1[:, ndim // 2]
    ).sum(axis=1) / float(fcsterr1.shape[0] - 1)
    ferrstd = np.sqrt(
        (fcsterr1**2).sum(axis=0) / float(fcsterr1.shape[0] - 1)
    )
    errcovmean = fcsterrcorr
    fcsterrcorr = fcsterrcorr / (ferrstd * ferrstd[ndim // 2])

    print(
        method,
        ncount,
        corrl,
        covinflate1,
        covinflate2,
        oberrstdev,
        np.sqrt(fcsterr.mean()),
        fstdev,
        np.sqrt(analerr.mean()),
        astdev,
        neig,
    )
