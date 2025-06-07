"""Lorenz 1996 model (with zonally varying damping).

Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
"""
import numpy as np


class L96:
    """Lorenz 1996 model with zonally varying damping."""

    def __init__(
        self,
        members=1,
        n=40,
        dt=0.05,
        diff_max=1.0,
        diff_min=1.0,
        F=8,
        deltaF=0,
        Fcorr=1.0,
        rs=None,
    ):
        self.n = n
        self.dt = dt
        self.diff_max = diff_max
        self.diff_min = diff_min
        self.F = F
        self.members = members
        self.deltaF = deltaF
        self.Fcorr = Fcorr

        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs

        self.x = F + 0.1 * self.rs.standard_normal(size=(members, n))
        self.blend = np.cos(np.linspace(0, np.pi, n)) ** 4
        self.xwrap = np.zeros((self.members, self.n + 3), float)

        if self.deltaF == 0:
            self.forcing = self.F
        else:
            self.forcing = self.rs.gamma(
                self.F / self.deltaF, self.deltaF, size=(self.members, self.n)
            )

    def shiftx(self):
        """Wrap the state for periodic boundary conditions."""
        xwrap = self.xwrap
        xwrap[:, 2 : self.n + 2] = self.x
        xwrap[:, 1] = self.x[:, -1]
        xwrap[:, 0] = self.x[:, -2]
        xwrap[:, -1] = self.x[:, 0]

        xm2 = xwrap[:, 0 : self.n]
        xm1 = xwrap[:, 1 : self.n + 1]
        xp1 = xwrap[:, 3 : self.n + 3]
        return xm2, xm1, xp1

    def dxdt(self):
        """Compute time derivative dx/dt."""
        xm2, xm1, xp1 = self.shiftx()
        damping = self.diff_min + (
            self.diff_max - self.diff_min
        ) * self.blend
        return (xp1 - xm2) * xm1 - damping * self.x + self.forcing

    def advance(self):
        """Advance the state by one time step using RK4."""
        if self.deltaF == 0:
            self.forcing = self.F
        else:
            deltaF_adjusted = self.deltaF / (1.0 - np.sqrt(self.Fcorr))
            forcing_new = (
                self.Fcorr * self.forcing
                + (1.0 - self.Fcorr)
                * self.rs.gamma(
                    self.F / deltaF_adjusted,
                    deltaF_adjusted,
                    size=(self.members, self.n),
                )
            )
            self.forcing = forcing_new

        h = self.dt
        hh = 0.5 * h
        h6 = h / 6.0
        x = self.x

        dx1 = self.dxdt()
        self.x = x + hh * dx1

        dx2 = self.dxdt()
        self.x = x + hh * dx2

        dx3 = self.dxdt()
        self.x = x + h * dx3

        dx4 = self.dxdt()
        self.x = x + h6 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)

        # Restore final state
        self.x = self.x

