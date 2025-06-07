import numpy as np
from scipy.linalg import (
    eigh,
    cho_solve,
    cho_factor,
    svd,
    pinvh
)


def symsqrt_psd(a, inv=False):
    """Symmetric square-root of a symmetric positive-definite matrix."""
    evals, eigs = eigh(a)
    symsqrt = (eigs * np.sqrt(np.maximum(evals, 0))).dot(eigs.T)
    if inv:
        inv_mat = (eigs * (1.0 / np.maximum(evals, 0.0))).dot(eigs.T)
        return symsqrt, inv_mat
    return symsqrt


def symsqrtinv_psd(a):
    """Inverse and inverse symmetric square-root of a symmetric
    positive-definite matrix."""
    evals, eigs = eigh(a)
    evals = np.where(evals < 1e-8, 1e-8, evals)
    symsqrtinv = (
        eigs * (1.0 / np.sqrt(np.maximum(evals, 0)))
    ).dot(eigs.T)
    inv_mat = (eigs * (1.0 / np.maximum(evals, 0))).dot(eigs.T)
    return symsqrtinv, inv_mat


def serial_ensrf(xmean, xprime, h, obs, oberrvar, covlocal, obcovlocal):
    """Serial Potter method."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    for nob, ob in zip(range(nobs), obs):
        # Forward operator
        hxprime = np.dot(xprime, h[nob])
        hxmean = np.dot(h[nob], xmean)

        # State-space update
        hxens = hxprime.reshape((nanals, 1))
        D = (hxens**2).sum() / (nanals - 1) + oberrvar
        gainfact = np.sqrt(D) / (np.sqrt(D) + np.sqrt(oberrvar))
        pbht = (xprime.T * hxens[:, 0]).sum(axis=1) / float(nanals - 1)
        kfgain = covlocal[nob, :] * pbht / D

        xmean = xmean + kfgain * (ob - hxmean)
        xprime = xprime - gainfact * kfgain * hxens

    return xmean, xprime


def serial_ensrf_modens(
    xmean, xprime, h, obs, oberrvar, covlocal, z
):
    """Serial Potter method with modulation ensemble."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    update_xprime = True
    if z is None:
        # Set ensemble to square root of localized Pb
        Pb = covlocal * np.dot(xprime.T, xprime) / (nanals - 1)
        evals, eigs = eigh(Pb)
        evals = np.where(evals > 1e-10, evals, 1e-10)
        nanals2 = eigs.shape[0]
        xprime2 = np.sqrt(nanals2 - 1) * (eigs * np.sqrt(evals)).T
    else:
        # Modulation ensemble
        neig = z.shape[0]
        nanals2 = neig * nanals
        xprime2 = np.zeros((nanals2, ndim), xprime.dtype)
        nanal2 = 0
        for j in range(neig):
            for nanal in range(nanals):
                xprime2[nanal2, :] = xprime[nanal, :] * z[neig - j - 1, :]
                nanal2 += 1
        xprime2 = np.sqrt(
            float(nanals2 - 1) / float(nanals - 1)
        ) * xprime2

    # Update xmean using full xprime2; update original xprime
    # using gain from full xprime2
    for nob, ob in zip(range(nobs), obs):
        hxprime = np.dot(xprime2, h[nob])
        hxprime_orig = np.dot(xprime, h[nob])
        hxmean = np.dot(h[nob], xmean)

        hxens = hxprime.reshape((nanals2, 1))
        hxens_orig = hxprime_orig.reshape((nanals, 1))
        D = (hxens**2).sum() / (nanals2 - 1) + oberrvar
        gainfact = np.sqrt(D) / (np.sqrt(D) + np.sqrt(oberrvar))
        pbht = (xprime2.T * hxens[:, 0]).sum(axis=1) / float(nanals2 - 1)
        kfgain = pbht / D

        xmean = xmean + kfgain * (ob - hxmean)
        xprime2 = xprime2 - gainfact * kfgain * hxens

        if not update_xprime:
            D = (hxens_orig**2).sum() / (nanals - 1) + oberrvar
            gainfact = np.sqrt(D) / (np.sqrt(D) + np.sqrt(oberrvar))
            pbht = (xprime.T * hxens_orig[:, 0]).sum(axis=1) / float(nanals - 1)
            kfgain = covlocal[nob, :] * pbht / D

        xprime = xprime - gainfact * kfgain * hxens_orig

    return xmean, xprime


def bulk_ensrf(xmean, xprime, h, obs, oberrvar, covlocal, denkf=False):
    """Bulk Potter method."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]
    R = oberrvar * np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar) * np.eye(nobs)

    Pb = np.dot(xprime.T, xprime) / (nanals - 1)
    Pb = covlocal * Pb

    D = np.dot(np.dot(h, Pb), h.T) + R
    if not denkf:
        Dsqrt, Dinv = symsqrt_psd(D, inv=True)
    else:
        Dinv = cho_solve(cho_factor(D), np.eye(nobs))

    kfgain = np.dot(np.dot(Pb, h.T), Dinv)
    if not denkf:
        tmp = Dsqrt + Rsqrt
        tmpinv = cho_solve(cho_factor(tmp), np.eye(nobs))
        gainfact = np.dot(Dsqrt, tmpinv)
        reducedgain = np.dot(kfgain, gainfact)
    else:
        reducedgain = 0.5 * kfgain

    xmean = xmean + np.dot(kfgain, obs - np.dot(h, xmean))

    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal])

    xprime = xprime - np.dot(reducedgain, hxprime.T).T
    return xmean, xprime


def bulk_enkf(xmean, xprime, h, obs, oberrvar, covlocal, rs):
    """Bulk EnKF method with perturbed observations."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]
    R = oberrvar * np.eye(nobs)
    Rsqrt = np.sqrt(oberrvar) * np.eye(nobs)

    Pb = np.dot(xprime.T, xprime) / (nanals - 1)
    Pb = covlocal * Pb

    D = np.dot(np.dot(h, Pb), h.T) + R
    Dinv = cho_solve(cho_factor(D), np.eye(nobs))
    kfgain = np.dot(np.dot(Pb, h.T), Dinv)
    # Apply localization to Kalman gain
    # Reduce the square localization matrix to the columns that correspond
    # to the observed state indices so that the shapes match (npts, nobs).
    obs_idx = np.argmax(h, axis=1)        # indices of observed variables
    kfgain = covlocal[:, obs_idx] * kfgain

    xmean = xmean + np.dot(kfgain, obs - np.dot(h, xmean))

    obnoise = np.sqrt(oberrvar) * rs.standard_normal((nanals, nobs))
    obnoise_var = ((obnoise - obnoise.mean(axis=0))**2).sum(axis=0) / (
        nanals - 1
    )
    obnoise = np.sqrt(oberrvar) * obnoise / np.sqrt(obnoise_var)

    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal]) + obnoise[nanal]

    xprime = xprime - np.dot(kfgain, hxprime[:, :, np.newaxis]).T.squeeze()
    return xmean, xprime


def etkf(xmean, xprime, h, obs, oberrvar):
    """ETKF (use only with full-rank ensemble, no localization)."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal])

    hxmean = np.dot(h, xmean)
    Rinv = (1.0 / oberrvar) * np.eye(nobs)
    YbRinv = np.dot(hxprime, Rinv)

    pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hxprime.T)
    pasqrt_inv, painv = symsqrtinv_psd(pa)

    kfgain = np.dot(xprime.T, np.dot(painv, YbRinv))
    enswts = np.sqrt(nanals - 1) * pasqrt_inv

    xmean = xmean + np.dot(kfgain, obs - hxmean)
    xprime = np.dot(enswts.T, xprime)
    return xmean, xprime


def getkf(xmean, xprime, h, obs, oberrvar):
    """GETKF (use only with full-rank ensemble, no localization)."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal])

    hxmean = np.dot(h, xmean)
    sqrtoberrvar_inv = 1.0 / np.sqrt(oberrvar)
    YbRsqrtinv = hxprime * sqrtoberrvar_inv

    u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver="gesvd")
    sp = s**2 + nanals - 1
    painv = (u * (1.0 / sp)).dot(u.T)

    kfgain = np.dot(
        xprime.T, np.dot(painv, YbRsqrtinv * sqrtoberrvar_inv)
    )
    xmean = xmean + np.dot(kfgain, obs - hxmean)

    reducedgain = np.dot(xprime.T, u) * (
        1.0 - np.sqrt((nanals - 1) / sp)
    )
    reducedgain = np.dot(
        reducedgain, (v.T / s).T
    ) * sqrtoberrvar_inv

    xprime = xprime - np.dot(reducedgain, hxprime.T).T
    return xmean, xprime


def getkf_modens(
    xmean, xprime, h, obs, oberrvar, covlocal, z
):
    """GETKF with modulated ensemble."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    if z is None:
        raise ValueError("z not specified")

    neig = z.shape[0]
    nanals2 = neig * nanals
    xprime2 = np.zeros((nanals2, ndim), xprime.dtype)
    nanal2 = 0

    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2, :] = xprime[nanal, :] * z[neig - j - 1, :]
            nanal2 += 1

    xprime2 = np.sqrt(
        float(nanals2 - 1) / float(nanals - 1)
    ) * xprime2

    hxprime = np.empty((nanals2, nobs), xprime2.dtype)
    hxprime_orig = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals2):
        hxprime[nanal] = np.dot(h, xprime2[nanal])
    for nanal in range(nanals):
        hxprime_orig[nanal] = np.dot(h, xprime[nanal])

    hxmean = np.dot(h, xmean)
    sqrtoberrvar_inv = 1.0 / np.sqrt(oberrvar)
    YbRsqrtinv = hxprime * sqrtoberrvar_inv

    u, s, v = svd(YbRsqrtinv, full_matrices=False, lapack_driver="gesvd")
    sp = s**2 + nanals2 - 1
    painv = (u * (1.0 / sp)).dot(u.T)

    kfgain = np.dot(
        xprime2.T,
        np.dot(painv, YbRsqrtinv * sqrtoberrvar_inv)
    )
    xmean = xmean + np.dot(kfgain, obs - hxmean)

    reducedgain = np.dot(xprime2.T, u) * (
        1.0 - np.sqrt((nanals2 - 1) / sp)
    )
    reducedgain = np.dot(
        reducedgain, (v.T / s).T
    ) * sqrtoberrvar_inv

    xprime = xprime - np.dot(reducedgain, hxprime_orig.T).T
    return xmean, xprime


def etkf_modens(
    xmean,
    xprime,
    h,
    obs,
    oberrvar,
    covlocal,
    z,
    rs=None,
    po=False,
    ss=False,
    adjust_obnoise=False,
    denkf=False
):
    """ETKF with modulated ensemble."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    if z is None:
        raise ValueError("z not specified")

    neig = z.shape[0]
    nanals2 = neig * nanals
    xprime2 = np.zeros((nanals2, ndim), xprime.dtype)
    nanal2 = 0

    for j in range(neig):
        for nanal in range(nanals):
            xprime2[nanal2, :] = xprime[nanal, :] * z[neig - j - 1, :]
            nanal2 += 1

    normfact = np.sqrt(float(nanals2 - 1) / float(nanals - 1))
    xprime2 = normfact * xprime2
    scalefact = normfact * z[-1]

    hxprime = np.empty((nanals2, nobs), xprime2.dtype)
    hxprime_orig = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals2):
        hxprime[nanal] = np.dot(h, xprime2[nanal])
    for nanal in range(nanals):
        hxprime_orig[nanal] = np.dot(h, xprime[nanal])

    hxmean = np.dot(h, xmean)
    YbRinv = np.dot(hxprime, (1.0 / oberrvar) * np.eye(nobs))

    pa = (nanals2 - 1) * np.eye(nanals2) + np.dot(YbRinv, hxprime.T)
    pasqrt_inv, painv = symsqrtinv_psd(pa)

    kfgain = np.dot(xprime2.T, np.dot(painv, YbRinv))
    xmean = xmean + np.dot(kfgain, obs - hxmean)

    if denkf:
        xprime = xprime - np.dot(0.5 * kfgain, hxprime_orig.T).T
    elif po:
        if rs is None:
            raise ValueError("must pass random state if po=True")

        obnoise = rs.standard_normal((nanals, nobs))
        obnoise = obnoise - obnoise.mean(axis=0)

        if adjust_obnoise:
            cxy = np.dot(obnoise, hxprime_orig.T)
            cxx = np.dot(hxprime_orig, hxprime_orig.T)
            cxxinv = pinvh(cxx)
            obnoise = obnoise - np.dot(
                np.dot(cxy, cxxinv), hxprime_orig
            )
            obnoise = obnoise - obnoise.mean(axis=0)

        obnoise = np.sqrt(
            oberrvar
            / (((obnoise**2).sum(axis=0) / (nanals - 1)).mean())
        ) * obnoise

        hxprime = obnoise + hxprime_orig
        xprime = xprime - np.dot(kfgain, hxprime.T).T
    elif ss:
        pasqrt_inv, painv = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2 - 1) * pasqrt_inv
        xprime_full = np.dot(enswts.T, xprime2)

        ranwts = rs.standard_normal((nanals, nanals2)) / np.sqrt(nanals2 - 1)
        ranwts_mean = ranwts.mean(axis=1)
        ranwts = ranwts - ranwts_mean[:, np.newaxis]
        ranwts_stdev = np.sqrt((ranwts**2).sum(axis=1))
        ranwts = ranwts / ranwts_stdev[:, np.newaxis]

        xprime = np.dot(ranwts, xprime_full)
        xprime = xprime - xprime.mean(axis=0)
    else:
        pasqrt_inv, _ = symsqrtinv_psd(pa)
        enswts = np.sqrt(nanals2 - 1) * pasqrt_inv
        xprime = np.dot(enswts[:, :nanals].T, xprime2) / scalefact

    return xmean, xprime


def letkf(
    xmean, xprime, h, obs, oberrvar, obcovlocal
):
    """LETKF (with observation localization)."""
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]

    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal])

    hxmean = np.dot(h, xmean)
    obcovlocal = np.where(obcovlocal < 1e-13, 1e-13, obcovlocal)

    xprime_prior = xprime.copy()
    xmean_prior = xmean.copy()
    ominusf = obs - np.dot(h, xmean_prior)

    for n in range(ndim):
        Rinv = np.diag(obcovlocal[n, :] / oberrvar)
        YbRinv = np.dot(hxprime, Rinv)
        pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hxprime.T)
        evals, eigs = eigh(pa)
        painv = np.dot(
            np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T
        )
        kfgain = np.dot(
            xprime_prior[:, n].T,
            np.dot(np.dot(painv, painv.T), YbRinv)
        )
        enswts = np.sqrt(nanals - 1) * painv

        xmean[n] = xmean[n] + np.dot(kfgain, ominusf)
        xprime[:, n] = np.dot(enswts.T, xprime_prior[:, n])

    return xmean, xprimes




# Dropout-EnKF method: Applies random dropout to Kalman gain computation for regularization.
def dropout_enkf(xmean, xprime, h, obs, oberrvar, covlocal, rs, dropout_rate=0.1):
    """
    Dropout-EnKF method: Applies random dropout to Kalman gain computation
    for regularization.
    
    Parameters:
        xmean (ndarray): Mean of the prior ensemble.
        xprime (ndarray): Anomaly matrix (ensemble perturbations).
        h (ndarray): Observation operator (as matrix).
        obs (ndarray): Observation vector.
        oberrvar (float): Observation error variance (scalar).
        covlocal (ndarray): Localization matrix.
        rs (np.random.RandomState): Random state generator.
        dropout_rate (float): Fraction of elements to zero-out in the gain.

    Returns:
        Updated xmean and xprime.
    """
    nanals, ndim = xprime.shape
    nobs = obs.shape[-1]
    R = oberrvar * np.eye(nobs)

    # Prior covariance
    Pb = np.dot(xprime.T, xprime) / (nanals - 1)
    Pb = covlocal * Pb

    D = np.dot(np.dot(h, Pb), h.T) + R
    Dinv = cho_solve(cho_factor(D), np.eye(nobs))

    kfgain = np.dot(np.dot(Pb, h.T), Dinv)
    # Apply localization to Kalman gain
    obs_idx = np.argmax(h, axis=1)        # indices of observed variables
    kfgain = covlocal[:, obs_idx] * kfgain
    # Apply dropout to Kalman gain
    dropout_mask = rs.binomial(1, 1 - dropout_rate, size=kfgain.shape)
    kfgain = kfgain * dropout_mask

    # Update mean
    xmean = xmean + np.dot(kfgain, obs - np.dot(h, xmean))

    # Update ensemble
    hxprime = np.empty((nanals, nobs), xprime.dtype)
    for nanal in range(nanals):
        hxprime[nanal] = np.dot(h, xprime[nanal])

    xprime = xprime - np.dot(kfgain, hxprime.T).T
    return xmean, xprime