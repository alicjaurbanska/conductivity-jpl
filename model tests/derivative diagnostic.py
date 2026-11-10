# =====================================================================
# Derivative diagnostics for the ionic splines
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

assert 'SEQ_SPLINES' in globals(), 'Run the main model cell first.'

def dIdT(spl, T, I, P_MPa=None):
    """d^2 lambda / (dI dT) for an IonSpline, Walden-aware.

    lambda_I = W_x * c / eta^alpha,  c = 1/(2 sqrt(I)) if SQRT_I else 1
    d/dT lambda_I = [W_xT * c - alpha * W_x * c * (deta/dT)/eta] / eta^alpha
    """
    T = np.atleast_1d(np.asarray(T, float))
    I = np.atleast_1d(np.asarray(I, float))
    x = spl._x(I)
    g   = spl._spline.ev(T, x, dx=0, dy=1)   # dW/dx
    gT  = spl._spline.ev(T, x, dx=1, dy=1)   # d2W/(dT dx)
    c = 1.0 / (2.0 * np.sqrt(I)) if SQRT_I else 1.0
    if WALDEN:
        P = spl.P_ref if P_MPa is None else P_MPa
        eta = VISC.eta_multi(T, P)
        val = (gT * c - spl.alpha * g * c * VISC.deta_dT(T, P) / eta) \
              / eta ** spl.alpha
    else:
        val = gT * c
    return np.where(spl._in_hull(T, I), val, np.nan)

# ---------- numeric cross-check of dIdT (finite differences) ----------
def _check_dIdT(spl, n=200, hT=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(spl.t_range[0] + 1, spl.t_range[1] - 1, n)
    I = np.exp(rng.uniform(np.log(max(1e-6, spl.I_range[0]) * 1.05),
                           np.log(spl.I_range[1] * 0.95), n))
    ok = spl._in_hull(T, I)
    T, I = T[ok], I[ok]
    if len(T) == 0:
        return np.nan
    ana = dIdT(spl, T, I)
    num = (spl.dI_val(T + hT, I) - spl.dI_val(T - hT, I)) / (2 * hT)
    good = np.isfinite(ana) & np.isfinite(num)
    scale = np.median(np.abs(ana[good])) or 1.0
    return np.nanmax(np.abs(ana[good] - num[good])) / scale

# ---------- derivative diagnostics table ----------
print('=== derivative diagnostics (dense grid, in-hull points only) ===')
rows = []
for name, s in SEQ_SPLINES.items():
    I_lo = max(1e-6, s.I_range[0])
    Ig = np.geomspace(I_lo, s.I_range[1], 200)
    Tg = np.linspace(s.t_range[0], s.t_range[1], 100)
    II, TT = np.meshgrid(Ig, Tg, indexing='ij')
    Tf, If = TT.ravel(), II.ravel()

    lam = s.ev_or_nan(Tf, If)
    dT  = s.dT(Tf, If)
    dI  = s.dI_val(Tf, If)
    d2  = s.d2I_val(Tf, If)

    fin = np.isfinite(lam)
    lam, dT_, dI_, d2_ = lam[fin], dT[fin], dI[fin], d2[fin]
    tc = dT_ / lam * 100.0  # % per K

    rows.append([name, len(lam), lam.min(),
                 (lam < 0).mean() * 100,
                 (dT_ > 0).mean() * 100,
                 np.percentile(tc, 1), np.percentile(tc, 99),
                 (dI_ < 0).mean() * 100,
                 (d2_ > 0).mean() * 100,
                 _check_dIdT(s)])
cols = ['ion', 'n_inhull', 'lam_min', '%lam<0', '%dT>0',
        'tcoef_p1 (%/K)', 'tcoef_p99 (%/K)', '%dI<0', '%d2I>0',
        'dIdT_fd_check']
print(pd.DataFrame(rows, columns=cols).round(3).to_string(index=False))
print('physics requires: lam>0, dT>0 (approx 1-3 %/K), dI<0, d2I>0 everywhere')
print('dIdT_fd_check = max |analytic - finite-diff| / typical scale (should be <<1)')

# ---------- 3D surface plots of lambda and its derivatives ----------
print('\n=== Plotting derivative diagnostic surfaces ===')
for name, s in SEQ_SPLINES.items():
    I_lo = max(1e-6, s.I_range[0])
    Ig = np.geomspace(I_lo, s.I_range[1], 60)
    Tg = np.linspace(s.t_range[0], s.t_range[1], 60)
    II, TT = np.meshgrid(Ig, Tg, indexing='ij')
    Tf, If = TT.ravel(), II.ravel()
    shape = II.shape

    panels = [
        (s.ev_or_nan(Tf, If).reshape(shape), r'$\lambda$ (>0)',            'viridis'),
        (s.dT(Tf, If).reshape(shape),        r'$d\lambda/dT$ (>0)',        'plasma'),
        (s.dI_val(Tf, If).reshape(shape),    r'$d\lambda/dI$ (<0)',        'coolwarm'),
        (s.d2I_val(Tf, If).reshape(shape),   r'$d^2\lambda/dI^2$ (>0)',    'magma'),
        (dIdT(s, Tf, If).reshape(shape),     r'$d^2\lambda/(dI\,dT)$',     'cividis'),
    ]

    fig = plt.figure(figsize=(22, 4))
    fig.suptitle(f'Ion: {name} - spline derivative diagnostics '
                 f'(Walden alpha={s.alpha:.3f}, P={s.P_ref} MPa)',
                 fontsize=14, y=1.05)
    for i, (Z, title, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 5, i, projection='3d')
        ax.plot_surface(TT, II, Z, cmap=cmap, edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel('T (K)')
        ax.set_ylabel(r'I$_{eff}$ (mol kg$^{-1}$)')
    plt.tight_layout()
    save_fig(fig, f'{name}_derivatives_diagnostic')
    plt.show()
    plt.close(fig)

print('\nDerivative diagnostics complete.')
