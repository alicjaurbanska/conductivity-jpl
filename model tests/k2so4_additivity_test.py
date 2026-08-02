# ==========================================================
# CELL 2 - K2SO4 HELD-OUT ADDITIVITY TEST (v7)
#   Empirical ion-surface model vs McCleskey (2012) baseline
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import stats as sstats

for _name in ('model', 'salt_dfs', 'spec_lookup', 'save_fig', 'FIG_DIR',
              'FIT_MODE', 'TESTS_ARE_NONCIRCULAR', 'REFERENCE_PRESSURE_MPA'):
    if _name not in globals():
        raise NameError(f"'{_name}' not found - run Cell 1 first.")
if 'mccleskey2012_conductivity' not in globals():
    raise NameError("mccleskey2012_conductivity not found - run the MC12 cell first.")

PCT_STABLE_MOLALITY = 0.01
MC12_T_VALID_C = (0.0, 95.0)

print("=== K2SO4 held-out additivity test (v7) ===")
if not TESTS_ARE_NONCIRCULAR:
    print("*** WARNING: FIT_MODE='joint' with JOINT_INCLUDE_ALL=True means "
          "K2SO4 was IN the training set. This test is IN-SAMPLE, not a "
          "held-out validation. Interpret accordingly. ***")
else:
    print(f"FIT_MODE='{FIT_MODE}': K2SO4 is genuinely held out (non-circular test).")
print("CAVEAT (shared submodel): both this model and the MC12 baseline use the "
      "McCleskey Table 1 KSO4- pair lambdas. The comparison is independent "
      "only in the free-ion channel; the pair-channel share of kappa is "
      "quantified below.")

# ==========================================================
# SHARED DIAGNOSTIC HELPERS
# ==========================================================

def fit_quality_stats(real, reconstructed):
    real = np.asarray(real, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    resid = reconstructed - real
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
    ss_res_11 = np.sum((reconstructed - real)**2)
    ss_tot = np.sum((real - real.mean())**2)
    r2_11 = 1 - ss_res_11/ss_tot if ss_tot > 0 else np.nan
    slope, intercept = np.polyfit(real, reconstructed, 1)
    return {'rmse': rmse, 'mae': mae, 'r2_vs_1to1_line': r2_11,
            'best_fit_slope': slope, 'best_fit_intercept': intercept, 'n': len(real)}

def print_fit_quality(label, real, reconstructed, units='mS/cm'):
    s = fit_quality_stats(real, reconstructed)
    print(f"\n  Fit quality ({label}):")
    print(f"    n = {s['n']}")
    print(f"    RMSE: {s['rmse']:.4f} {units}   MAE: {s['mae']:.4f} {units}")
    print(f"    R-squared vs perfect 1:1 line: {s['r2_vs_1to1_line']:.4f}")
    print(f"    Best-fit: reconstructed = {s['best_fit_slope']:.4f} x real + {s['best_fit_intercept']:.4f}")
    return s

def diagnose_error_structure(label, T_vals, M_vals, pct_dev):
    T_vals = np.asarray(T_vals, dtype=float)
    M_vals = np.asarray(M_vals, dtype=float)
    pct_dev = np.asarray(pct_dev, dtype=float)
    print(f"\n  Error structure diagnosis ({label}):")
    if len(T_vals) < 4:
        print("    Too few points for statistical breakdown.")
        return
    if len(set(M_vals)) > 1:
        r_m, p_m = sstats.pearsonr(M_vals, pct_dev)
        print(f"    dev vs molality:    r={r_m:+.3f} (p={p_m:.4f})")
    if len(set(T_vals)) > 1:
        r_t, p_t = sstats.pearsonr(T_vals, pct_dev)
        print(f"    dev vs temperature: r={r_t:+.3f} (p={p_t:.4f})")

# ==========================================================
# 1. SELECT USABLE K2SO4 POINTS (ambient only; speciated prediction)
# ==========================================================

d_all = salt_dfs['K2SO4']
n_total = len(d_all)
n_pressure = int((~d_all['Ambient']).sum())
d = d_all[d_all['Ambient']].reset_index(drop=True)
print(f"\nK2SO4: {n_total} rows total; {n_pressure} non-ambient pressure row(s) "
      f"excluded here (they belong to the staged pressure test); "
      f"{len(d)} ambient rows enter this test.")

T_K = d['temp_K'].values
T_C = d['Temperature'].values
M = d['Molality'].values
I_eff = d['I'].values
kappa_meas = d['Conductivity'].values
Source = d['Source'].values

# ---- speciated composition straight from the merged library columns ----
free = {}
for ion, col in ION_FREE_COL.items():
    if col in d.columns:
        free[ion] = np.nan_to_num(d[col].values)
pairs = {}
for col, key in PAIR_LAMBDA_FOR_COL.items():
    if col in d.columns:
        pairs[key] = np.nan_to_num(d[col].values)
if 'm_pair_MgSO4' in d.columns:
    pairs['MgSO4'] = np.nan_to_num(d['m_pair_MgSO4'].values)

comp = {'I_eff': I_eff, 'free_molality': free, 'pair_molality': pairs,
        'speciation_source': SPECIATION_DB, 'sample_id': 'K2SO4 held-out'}

kappa_model, diag = model.predict_conductivity(
    comp, T_K, REFERENCE_PRESSURE_MPA, extrap_mode='linear',
    return_diagnostics=True)
kappa_model = np.atleast_1d(kappa_model)

# ---- coverage report from diagnostics ----
n_extrap = np.zeros(len(d), dtype=bool)
n_oob = np.zeros(len(d), dtype=bool)
for sp, flag in diag['flags']['out_of_hull'].items():
    k = int(np.sum(flag))
    if k:
        print(f"  {sp}: {k} point(s) outside the ion-surface hull (unrecoverable).")
    n_oob |= flag
for sp, flag in diag['flags']['extrapolated'].items():
    k = int(np.sum(flag))
    if k:
        print(f"  {sp}: {k} point(s) recovered via cold linear extrapolation.")
    n_extrap |= flag

usable = np.isfinite(kappa_model)
print(f"Usable: {int(usable.sum())}/{len(d)} points "
      f"({int(n_extrap[usable].sum())} of them use cold-extrapolated ion lambdas).")
if usable.sum() == 0:
    raise RuntimeError("No usable K2SO4 points inside coverage.")

# ---- pair-channel share (shared-submodel circularity quantifier) ----
pair_kappa = diag['per_pair_kappa'].get('KSO4', np.zeros_like(kappa_model))
pair_frac = np.where(kappa_model > 0, pair_kappa / kappa_model, np.nan)
pf = pair_frac[usable]
print(f"\nKSO4- pair share of predicted kappa (shared MC12 submodel): "
      f"min={np.nanmin(pf):.1%}, median={np.nanmedian(pf):.1%}, "
      f"max={np.nanmax(pf):.1%}. The model-vs-MC12 comparison is independent "
      f"in the remaining (free-ion) share.")

# ==========================================================
# 2. McCLESKEY (2012) BASELINE - same speciation, same I_eff
# ==========================================================

mc12_species = {'K+': d['m_free_K'].values,
                'SO4-2': d['m_free_SO4'].values}
if 'm_pair_KSO4' in d.columns:
    mc12_species['KSO4-'] = np.nan_to_num(d['m_pair_KSO4'].values)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mc12_res = mccleskey2012_conductivity(
        mc12_species, T_C, I_eff_molkg=I_eff,
        speciation_source=SPECIATION_DB, validate_range=False)
kappa_mc12 = np.atleast_1d(mc12_res.sigma_mScm)

in_mc12_range = (T_C >= MC12_T_VALID_C[0]) & (T_C <= MC12_T_VALID_C[1]) & (I_eff <= 1.0)
n_out_range = int((~in_mc12_range & usable).sum())
if n_out_range:
    print(f"\nMC12 validity note: {n_out_range} usable point(s) lie outside the "
      f"published MC12 range (0-95 C, I <= 1 mol/kg); MC12 is extrapolating "
      f"its Table 1 polynomials there. Stats reported both ways below.")

# ==========================================================
# 3. DEVIATION METRICS - both models
# ==========================================================

u = usable
pct_dev_model = 100.0 * (kappa_model[u] - kappa_meas[u]) / kappa_meas[u]
pct_dev_mc12 = 100.0 * (kappa_mc12[u] - kappa_meas[u]) / kappa_meas[u]
Tu_K, Tu_C, Mu, Iu, Su = T_K[u], T_C[u], M[u], I_eff[u], Source[u]
Zu = kappa_meas[u]
stable = Mu > PCT_STABLE_MOLALITY
in_rng_u = in_mc12_range[u]

def _dev_line(name, dev, mask):
    if mask.sum() == 0:
        print(f"  {name}: no points in this subset.")
        return
    print(f"  {name}: mean signed {np.mean(dev[mask]):+.2f}%   "
          f"mean abs {np.mean(np.abs(dev[mask])):.2f}%   "
          f"max abs {np.max(np.abs(dev[mask])):.2f}%   (n={int(mask.sum())})")

print(f"\n--- Deviation statistics (m > {PCT_STABLE_MOLALITY} mol/kg) ---")
_dev_line('This model     ', pct_dev_model, stable)
_dev_line('MC12 baseline  ', pct_dev_mc12, stable)
print(f"--- Same, restricted to MC12 validity range ---")
_dev_line('This model     ', pct_dev_model, stable & in_rng_u)
_dev_line('MC12 baseline  ', pct_dev_mc12, stable & in_rng_u)

print("\n--- Per-source mean abs deviation (this model | MC12) ---")
for src in sorted(set(Su)):
    m_src = (Su == src) & stable
    if m_src.any():
        print(f"  {src:16s}: {np.mean(np.abs(pct_dev_model[m_src])):6.2f}% | "
              f"{np.mean(np.abs(pct_dev_mc12[m_src])):6.2f}%  (n={int(m_src.sum())})")

print_fit_quality('This model: K2SO4 reconstructed vs empirical', Zu, kappa_model[u])
print_fit_quality('MC12 baseline: K2SO4 vs empirical', Zu, kappa_mc12[u])
diagnose_error_structure('This model', Tu_K, Mu, pct_dev_model)
diagnose_error_structure('MC12 baseline', Tu_K, Mu, pct_dev_mc12)

# ==========================================================
# 4. DIAGNOSTIC FIGURES
# ==========================================================
unique_sources = sorted(set(Su))
markers = ['o', 's', '^', 'D', 'v', 'P']
MODEL_COLOR, MC12_COLOR = 'tab:blue', 'tab:red'

# --- deviation vs molality, both models ---
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.axhspan(-10, 10, color='0.85', alpha=0.6, zorder=0, label=r'$\pm$10%')
for i, src in enumerate(unique_sources):
    mask = Su == src
    ax.scatter(Mu[mask], pct_dev_model[mask], s=30, marker=markers[i % len(markers)],
               color=MODEL_COLOR, zorder=3,
               label=f'{src} (this model)' if i == 0 else None)
    ax.scatter(Mu[mask], pct_dev_mc12[mask], s=30, marker=markers[i % len(markers)],
               facecolors='none', edgecolors=MC12_COLOR, zorder=3,
               label=f'{src} (MC12)' if i == 0 else None)
ax.axhline(0, color='black', lw=0.8)
ax.set_xscale('log')
ax.set_xlabel(r'K$_2$SO$_4$ molality (mol kg$^{-1}$)')
ax.set_ylabel('Signed deviation (%)')
ax.set_title(r'K$_2$SO$_4$ held-out: deviation vs molality (filled = this model, open = MC12)')
ax.legend()
save_fig(fig, 'k2so4_deviation_vs_molality')
plt.show()

# --- deviation vs temperature, both models ---
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.axhspan(-10, 10, color='0.85', alpha=0.6, zorder=0, label=r'$\pm$10%')
ax.scatter(Tu_K, pct_dev_model, s=28, color=MODEL_COLOR, zorder=3, label='This model')
ax.scatter(Tu_K, pct_dev_mc12, s=28, facecolors='none', edgecolors=MC12_COLOR,
           zorder=3, label='MC12 baseline')
ax.axvline(MC12_T_VALID_C[1] + 273.15, color=MC12_COLOR, ls=':', lw=1,
           label='MC12 validity edge (95 C)')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Signed deviation (%)')
ax.set_title(r'K$_2$SO$_4$ held-out: deviation vs temperature')
ax.legend()
save_fig(fig, 'k2so4_deviation_vs_temperature')
plt.show()

# --- parity, both models ---
fig, ax = plt.subplots(figsize=(5.4, 5.4))
lims = [0, max(Zu.max(), np.nanmax(kappa_model[u]), np.nanmax(kappa_mc12[u])) * 1.05]
ax.plot(lims, lims, ls='--', color='gray', lw=1, label='1:1 line')
ax.scatter(Zu, kappa_model[u], s=28, color=MODEL_COLOR, zorder=3, label='This model')
ax.scatter(Zu, kappa_mc12[u], s=28, facecolors='none', edgecolors=MC12_COLOR,
           zorder=3, label='MC12 baseline')
s_m = fit_quality_stats(Zu, kappa_model[u])
s_b = fit_quality_stats(Zu, kappa_mc12[u])
ax.text(0.05, 0.95,
        f"This model: RMSE={s_m['rmse']:.3f}, $R^2$={s_m['r2_vs_1to1_line']:.4f}\n"
        f"MC12:       RMSE={s_b['rmse']:.3f}, $R^2$={s_b['r2_vs_1to1_line']:.4f}\n"
        f"n = {s_m['n']}",
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', fc='white', ec='black', lw=0.6))
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'Empirical conductivity (mS cm$^{-1}$)')
ax.set_ylabel(r'Predicted conductivity (mS cm$^{-1}$)')
ax.set_title(r'K$_2$SO$_4$: predicted vs empirical')
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=8)
save_fig(fig, 'k2so4_parity')
plt.show()

# --- structure of the error in (T, m) space, this model ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
vmax = max(np.max(np.abs(pct_dev_model)), 1.0)
sc = ax.scatter(Tu_K, Mu, c=pct_dev_model, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                s=60, edgecolors='black', linewidths=0.4)
plt.colorbar(sc, ax=ax, label='Signed deviation (%)')
ext_mask = n_extrap[u]
if ext_mask.any():
    ax.scatter(Tu_K[ext_mask], Mu[ext_mask], facecolors='none', edgecolors='lime',
               s=110, linewidths=1.2, label='cold-extrapolated ion lambda')
    ax.legend(fontsize=8)
ax.set_yscale('log')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel(r'K$_2$SO$_4$ molality (mol kg$^{-1}$)')
ax.set_title(r'K$_2$SO$_4$: structure of the additivity error (this model)')
save_fig(fig, 'k2so4_deviation_map')
plt.show()

# --- isotherms: data vs both model lines ---
fig, ax = plt.subplots(figsize=(6.8, 4.6))
temps = np.unique(np.round(Tu_K, 1))
cmap = plt.get_cmap('plasma')
m_grid = np.geomspace(Mu.min(), Mu.max(), 120)
for i, tval in enumerate(temps):
    color = cmap(i / max(len(temps) - 1, 1))
    mask = np.isclose(Tu_K, tval, atol=0.1)
    ax.scatter(Mu[mask], Zu[mask], color=color, s=26, zorder=3)
    T_line = np.full_like(m_grid, tval)
    z_line = np.atleast_1d(model.predict_from_speciation(
        'K2SO4', T_line, m_grid, extrap_mode='linear'))
    ok = np.isfinite(z_line)
    ax.plot(m_grid[ok], z_line[ok], color=color, lw=1.3, label=f'{tval:.0f} K')
    # MC12 line at the same speciated grid
    spc = {'K+': np.nan_to_num(spec_lookup.get('K2SO4', T_line, m_grid, 'm_free_K')),
           'SO4-2': np.nan_to_num(spec_lookup.get('K2SO4', T_line, m_grid, 'm_free_SO4')),
           'KSO4-': np.nan_to_num(spec_lookup.get('K2SO4', T_line, m_grid, 'm_pair_KSO4'))}
    I_line = spec_lookup.get('K2SO4', T_line, m_grid, 'I_eff')
    okI = np.isfinite(I_line)
    if okI.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            z_mc = mccleskey2012_conductivity(
                {k: v[okI] for k, v in spc.items()}, tval - 273.15,
                I_eff_molkg=I_line[okI], validate_range=False).sigma_mScm
        ax.plot(m_grid[okI], z_mc, color=color, lw=1.0, ls=':')
ax.set_xscale('log')
ax.set_xlabel(r'K$_2$SO$_4$ molality (mol kg$^{-1}$)')
ax.set_ylabel(r'Conductivity (mS cm$^{-1}$)')
ax.set_title(r'K$_2$SO$_4$ isotherms: data vs this model (solid) vs MC12 (dotted)')
ax.legend(title='T', ncol=2, fontsize=8)
save_fig(fig, 'k2so4_isotherms')
plt.show()

# --- deviation histograms, both models ---
fig, ax = plt.subplots(figsize=(5.8, 3.8))
bins = np.linspace(min(pct_dev_model[stable].min(), pct_dev_mc12[stable].min()),
                   max(pct_dev_model[stable].max(), pct_dev_mc12[stable].max()), 18)
ax.hist(pct_dev_model[stable], bins=bins, color=MODEL_COLOR, alpha=0.65,
        edgecolor='black', linewidth=0.4, label='This model')
ax.hist(pct_dev_mc12[stable], bins=bins, color=MC12_COLOR, alpha=0.45,
        edgecolor='black', linewidth=0.4, label='MC12 baseline')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Signed deviation (%)')
ax.set_ylabel('Count')
ax.set_title(rf'K$_2$SO$_4$ deviation distribution (m > {PCT_STABLE_MOLALITY})')
ax.legend()
save_fig(fig, 'k2so4_deviation_histogram')
plt.show()

# ==========================================================
# 5. BRIEF INTERPRETATION SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("SUMMARY (easy read)")
print("=" * 60)
mae_m = np.mean(np.abs(pct_dev_model[stable]))
mae_b = np.mean(np.abs(pct_dev_mc12[stable]))
bias_m = np.mean(pct_dev_model[stable])
bias_b = np.mean(pct_dev_mc12[stable])
better = 'This model' if mae_m < mae_b else 'MC12 baseline'
print(f"1. Held-out K2SO4, ambient rows, m > {PCT_STABLE_MOLALITY} mol/kg:")
print(f"     this model: MAE {mae_m:.2f}%, bias {bias_m:+.2f}%")
print(f"     MC12:       MAE {mae_b:.2f}%, bias {bias_b:+.2f}%")
print(f"     -> {better} is closer overall (difference {abs(mae_m - mae_b):.2f} points).")
if (stable & in_rng_u).sum() and (~in_rng_u & stable).sum():
    mae_m_in = np.mean(np.abs(pct_dev_model[stable & in_rng_u]))
    mae_b_in = np.mean(np.abs(pct_dev_mc12[stable & in_rng_u]))
    mae_m_out = np.mean(np.abs(pct_dev_model[stable & ~in_rng_u]))
    mae_b_out = np.mean(np.abs(pct_dev_mc12[stable & ~in_rng_u]))
    print(f"2. Inside MC12 validity (0-95 C): this model {mae_m_in:.2f}% vs MC12 {mae_b_in:.2f}%.")
    print(f"   Outside (MC12 extrapolating):  this model {mae_m_out:.2f}% vs MC12 {mae_b_out:.2f}%.")
print(f"3. Circularity caveats: {'test is IN-SAMPLE (joint fit trained on K2SO4)' if not TESTS_ARE_NONCIRCULAR else 'K2SO4 never entered training'};"
      f" both models share the KSO4- pair lambdas (pair share of kappa: "
      f"median {np.nanmedian(pf):.1%}), so the independent part of the "
      f"comparison is the free-ion channel.")
print(f"4. Coverage: {int(usable.sum())}/{len(d)} ambient points usable; "
      f"{int(n_extrap[u].sum())} rely on cold-extrapolated ion lambdas "
      f"(marked green in the deviation map); {n_pressure} pressure rows deferred "
      f"to the staged pressure test.")
