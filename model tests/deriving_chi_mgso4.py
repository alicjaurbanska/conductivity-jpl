# ==========================================================
FIT CHI (effective conductivity of associated MgSO4)
# kappa = sum(m_i * lambda_i) + chi * m_pair_MgSO4 * (lambda_Mg + lambda_SO4)
# Frozen: ion splines, Walden alphas, speciation (current SPECIATION_DB).
# Fit: binary MgSO4 only, ambient rows only, mixtures excluded.
# ==========================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt

CHI_EXTRAP_MODE = 'linear'   # cold-side known-ion extrapolation only
CHI_MIN_S_FRAC  = 0.02       # per-point chi_i only shown where the pair term
                             # could matter (s > 2% of measured kappa)

def _lam_with_cold_extrap(spl, T, I, P):
    lam = np.atleast_1d(spl.ev_or_nan(T, I, P_MPa=P)).astype(float)
    bad = ~np.isfinite(lam)
    if bad.any():
        lam[bad] = spl.ev_extrap(T[bad], I[bad], mode=CHI_EXTRAP_MODE, P_MPa=P[bad])
    return lam

print("=" * 70)
print(f"CHI FIT on binary MgSO4 (speciation = {SPECIATION_DB}, splines frozen)")
print("=" * 70)

d_all = salt_dfs['MgSO4'].copy()
d = d_all[d_all['Ambient']].copy()
print(f"MgSO4 pool: {len(d_all)} rows total, {len(d)} ambient rows used for the fit "
      f"({len(d_all) - len(d)} pressure rows held aside for a frozen-chi check).")

def evaluate_mgso4(dd):
    T = dd['temp_K'].values
    I = dd['I'].values
    P = dd['P_state_MPa'].values
    kappa_meas = dd['Conductivity'].values
    # base: all free charged species from the speciation columns
    kappa_base = np.zeros_like(T)
    for ion, col in ION_FREE_COL.items():
        if col not in dd.columns:
            continue
        m = np.nan_to_num(dd[col].values.astype(float))
        if not np.any(m > PREDICT_MIN_SPECIES_MOLALITY):
            continue
        kappa_base = kappa_base + m * _lam_with_cold_extrap(model.ion(ion), T, I, P)
    lam_mg  = _lam_with_cold_extrap(model.ion('Mg'),  T, I, P)
    lam_so4 = _lam_with_cold_extrap(model.ion('SO4'), T, I, P)
    m_pair  = np.nan_to_num(dd['m_pair_MgSO4'].values.astype(float))
    s = m_pair * (lam_mg + lam_so4)          # pair channel, mS/cm at chi=1
    ok = np.isfinite(kappa_base) & np.isfinite(s) & (kappa_meas > 0)
    return kappa_meas, kappa_base, s, ok

kappa_meas, kappa_base, s_chan, ok = evaluate_mgso4(d)
n_excl = int((~ok).sum())
if n_excl:
    I_excl = d['I'].values[~ok]
    print(f"Excluded {n_excl}/{len(d)} ambient rows with no finite prediction "
          f"(out-of-hull, I_eff {I_excl.min():.3f}-{I_excl.max():.3f}; "
          f"SO4 hull ceiling ~0.6 is the usual cause).")

km, kb, s = kappa_meas[ok], kappa_base[ok], s_chan[ok]
d_ok = d[ok].copy()

# ---- closed-form least squares in RELATIVE space ----------------------
r = (km - kb) / km          # relative missing conductivity
x = s / km                  # relative pair channel at chi=1
chi_hat = float(np.sum(x * r) / np.sum(x * x))
resid = r - chi_hat * x
chi_se = float(np.sqrt(np.sum(resid**2) / max(len(r) - 1, 1) / np.sum(x * x)))
print(f"\nchi = {chi_hat:.4f} +/- {chi_se:.4f}  (n = {len(r)})")
if not (0.0 <= chi_hat <= 1.0):
    print("NOTE: chi outside [0, 1] - the pair channel is compensating for "
          "something other than outer-sphere transport. Interpret with care.")
CHI_MGSO4 = chi_hat   # exported for Cell B

# ---- before/after metrics --------------------------------------------
def _metrics(pred, meas):
    dev = (pred - meas) / meas
    return float(np.mean(dev)), float(np.sqrt(np.mean(dev**2)))

bias0, rmse0 = _metrics(kb, km)
bias1, rmse1 = _metrics(kb + chi_hat * s, km)
print(f"\nBinary MgSO4, ambient rows (relative units):")
print(f"  chi = 0     : bias = {bias0:+.3%}, RMSE = {rmse0:.3%}")
print(f"  chi = {chi_hat:.3f}: bias = {bias1:+.3%}, RMSE = {rmse1:.3%}")
print("\nPer source:")
for src in sorted(d_ok['Source'].unique()):
    m = d_ok['Source'].values == src
    b0, r0 = _metrics(kb[m], km[m])
    b1, r1 = _metrics(kb[m] + chi_hat * s[m], km[m])
    print(f"  {src:14s} (n={int(m.sum()):3d}): chi=0 bias {b0:+.2%} RMSE {r0:.2%} "
          f"-> fitted bias {b1:+.2%} RMSE {r1:.2%}")

# ---- constancy check: per-point chi where the channel is resolvable ---
strong = x > CHI_MIN_S_FRAC
chi_i = np.where(strong, r / np.where(x > 0, x, np.nan), np.nan)
print(f"\nPer-point chi_i (only where pair channel > {CHI_MIN_S_FRAC:.0%} of kappa; "
      f"n = {int(strong.sum())}): median = {np.nanmedian(chi_i):.3f}, "
      f"IQR = {np.nanpercentile(chi_i, 25):.3f}-{np.nanpercentile(chi_i, 75):.3f}")

# ---- frozen-chi check on non-ambient MgSO4 rows ------------------------
d_p = d_all[~d_all['Ambient']].copy()
if len(d_p):
    km_p, kb_p, s_p, ok_p = evaluate_mgso4(d_p)
    if ok_p.any():
        b0, r0 = _metrics(kb_p[ok_p], km_p[ok_p])
        b1, r1 = _metrics(kb_p[ok_p] + chi_hat * s_p[ok_p], km_p[ok_p])
        print(f"\nFrozen-chi check on pressure rows (speciation off-plane, "
              f"viscosity channel only; n={int(ok_p.sum())}):")
        print(f"  chi=0: bias {b0:+.3%} RMSE {r0:.3%} -> fitted: bias {b1:+.3%} RMSE {r1:.3%}")

# ---- deviation plots ---------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
dev0 = 100 * (kb - km) / km
dev1 = 100 * (kb + chi_hat * s - km) / km
for ax, xv, xl in [(axes[0, 0], d_ok['Molality'].values, 'molality (mol/kg)'),
                   (axes[0, 1], d_ok['temp_K'].values - 273.15, 'T (C)')]:
    for src in sorted(d_ok['Source'].unique()):
        m = d_ok['Source'].values == src
        ax.scatter(xv[m], dev0[m], s=14, marker='o', alpha=0.5, label=f'{src} chi=0')
        ax.scatter(xv[m], dev1[m], s=14, marker='x', label=f'{src} fitted')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(xl); ax.set_ylabel('deviation (%)')
    ax.legend(fontsize=6)
axes[0, 0].set_title(f'MgSO4 deviation, chi=0 (o) vs chi={chi_hat:.3f} (x)')
ax = axes[1, 0]
ax.scatter(d_ok['I'].values[strong], chi_i[strong], s=14, c=d_ok['temp_K'].values[strong])
ax.axhline(chi_hat, color='r', lw=1, label=f'fitted chi={chi_hat:.3f}')
ax.set_xlabel('I_eff (mol/kg)'); ax.set_ylabel('per-point chi_i')
ax.set_title('chi constancy check (color = T)'); ax.legend(fontsize=8)
ax = axes[1, 1]
ax.scatter(km, kb, s=14, alpha=0.5, label='chi=0')
ax.scatter(km, kb + chi_hat * s, s=14, marker='x', label='fitted')
lim = [0, max(km.max(), (kb + chi_hat * s).max()) * 1.05]
ax.plot(lim, lim, 'k-', lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('measured kappa (mS/cm)'); ax.set_ylabel('predicted kappa (mS/cm)')
ax.set_title('parity'); ax.legend(fontsize=8)
fig.tight_layout(); save_fig(fig, 'chi_fit_mgso4'); plt.show()

print(f"\nCHI_MGSO4 = {CHI_MGSO4:.4f} stored for the mixture test cell.")
