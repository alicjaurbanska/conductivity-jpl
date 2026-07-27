# ==========================================================
# ADDITIVITY TESTS v3 - 2x2 ATTRIBUTION:
#   fitting strategy (bivariate splines vs McCleskey polynomials)
#     x speciation engine (SUPCRT vs WATEQ4F)
# ==========================================================
# Run AFTER Cell 1 (v5.2+) in the same notebook: uses Cell 1's namespace
# (model, splines, salt_dfs, spec_lookup, SpeciationLookup, SPECIATION_DB,
# subtract_ion_molar, IonConductivityModel, mc12_pair_lambda, save_fig,
# _data, LIBRARY_FILE, DATA_FROM_GITHUB, GITHUB_RAW, ...).
#
# Colab: upload BOTH speciation_supcrt.csv and speciation_wateq4f.csv (or
# have them on GitHub), then:
#   import requests
#   exec(requests.get('https://raw.githubusercontent.com/alicjaurbanska/'
#        'conductivity-jpl/main/model%20tests/additivity_tests.py').text)
#
# THE 2x2 DESIGN. Each test salt is scored on the same data with up to
# four predictors:
#   framework + native engine   (splines as trained by Cell 1)
#   framework + other engine *  (prediction-time engine swap ONLY - the
#                                splines were TRAINED with the native
#                                engine, so this cell is marked with * as
#                                inconsistent-with-training. For the fully
#                                consistent version, rerun Cell 1 with
#                                SPECIATION_DB set to the other engine and
#                                run these tests again.)
#   McCleskey + supcrt          (his polynomials, SUPCRT free ions/I_eff)
#   McCleskey + wateq4f         (his polynomials, WATEQ4F - his native
#                                engine, i.e. the method as published)
#
# ATTRIBUTION LOGIC:
#   compare framework vs McCleskey WITHIN one engine  -> fitting-strategy effect
#   compare supcrt vs wateq4f WITHIN one model        -> speciation-engine effect
#   If both models miss a salt by the same offset under both engines,
#   neither fitting nor engine is the culprit - look at the shared physics
#   (e.g. pair conduction treatment).
#
# TESTS (all ambient): K2SO4, CaSO4, MgSO4 (non-circular, held out),
# CaCl2 (in-sample consistency), Na2SO4 via chain B (SO4 refit from K2SO4).
# Adams & Hall rows are relative-only and cannot enter ambient tests.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sstats
from urllib.parse import quote

print("=" * 72)
print("ADDITIVITY TESTS v3: splines vs McCleskey  x  SUPCRT vs WATEQ4F")
print("=" * 72)

TEST_EXTRAP_MODE = 'linear'

NATIVE_DB = SPECIATION_DB
ALT_DB = 'wateq4f' if NATIVE_DB == 'supcrt' else 'supcrt'
ALT_FILE = f'speciation_{ALT_DB}.csv'
print(f"Framework splines were TRAINED with: {NATIVE_DB}. "
      f"'framework+{ALT_DB}*' below is a prediction-time engine swap only.")

def _try_load_lookup(fname):
    for path in ([fname] if not DATA_FROM_GITHUB else
                 [fname, GITHUB_RAW + quote('speciation databases/' + fname)]):
        try:
            return SpeciationLookup(pd.read_csv(path))
        except Exception:
            continue
    return None

spec_lookup_native = spec_lookup
spec_lookup_alt = _try_load_lookup(ALT_FILE)
if spec_lookup_alt is None:
    print(f"WARNING: {ALT_FILE} not found locally or on GitHub - running "
          f"WITHOUT the {ALT_DB} column of the 2x2.")
ENGINES = {NATIVE_DB: spec_lookup_native}
if spec_lookup_alt is not None:
    ENGINES[ALT_DB] = spec_lookup_alt

def _spec_get(lookup, salt, T_K, m, col):
    """lookup.get with graceful NaN when the table lacks the salt/column."""
    try:
        return lookup.get(salt, T_K, m, col)
    except KeyError:
        return np.full(np.atleast_1d(np.asarray(T_K, float)).shape, np.nan)

def _has_salt(lookup, salt):
    return lookup is not None and salt in lookup.interp2d

def make_model_with(lookup, base=None):
    """Same fitted splines, different speciation engine."""
    mm = IonConductivityModel(speciation=lookup)
    mm.splines = dict((base or model).splines)
    return mm

# ==========================================================
# McCLESKEY (2012) TABLE 1
# lam_i(T, I) = lam0(T) - A(T) sqrt(I)/(1 + B sqrt(I)); T in Celsius
# ==========================================================
MC_COEFFS = {
    #         lam0: (a, b, c)               A: (p, q, r)                 B
    'K':   ((0.003046, 1.261,  40.70), (0.00535, 0.9316,  22.59), 1.5),
    'Na':  ((0.003763, 0.8770, 26.23), (0.00027, 1.141,   32.07), 1.7),
    'Ca':  ((0.009645, 1.984,  62.28), (0.03174, 2.334,  132.3),  2.8),
    'Mg':  ((0.01068,  1.695,  57.16), (0.02453, 1.915,   80.50), 2.1),
    'NH4': ((0.003341, 1.285,  39.04), (0.00132, 0.6070,  11.19), 0.3),
    'Cl':  ((0.003817, 1.337,  40.99), (0.00613, 0.9469,  22.01), 1.5),
    'SO4': ((0.01037,  2.838,  82.37), (0.03324, 5.889,  193.5),  2.6),
}

def mccleskey_lambda(ion, I, T_C):
    (a, b, c), (p, q, r), B = MC_COEFFS[ion]
    T_C = np.asarray(T_C, dtype=float)
    sqrt_I = np.sqrt(np.asarray(I, dtype=float))
    lam0 = a * T_C**2 + b * T_C + c
    A = p * T_C**2 + q * T_C + r
    return lam0 - A * sqrt_I / (1.0 + B * sqrt_I)

def mccleskey_predict(salt, T_K, molality, lookup):
    """McCleskey ionic additivity fed the given speciation engine: free-ion
    molalities, I_eff, and his charged-pair lambdas. NaN where the engine
    lacks the salt or an ion lacks Table 1 coefficients."""
    T_K = np.atleast_1d(np.asarray(T_K, dtype=float))
    m = np.atleast_1d(np.asarray(molality, dtype=float))
    T_C = T_K - 273.15
    stoich = IonConductivityModel.SALT_STOICHIOMETRY[salt]
    if any(ion not in MC_COEFFS for ion in stoich) or not _has_salt(lookup, salt):
        return np.full(T_K.shape, np.nan)
    I = _spec_get(lookup, salt, T_K, m, 'I_eff')
    total = np.zeros_like(T_K, dtype=float)
    for ion in stoich:
        m_free = _spec_get(lookup, salt, T_K, m, ION_FREE_COL[ion])
        total = total + m_free * mccleskey_lambda(ion, I, T_C)
    if PAIR_CONDUCTION and salt in SALT_PAIR_TERMS:
        for col in SALT_PAIR_TERMS[salt]:
            m_pair = np.nan_to_num(_spec_get(lookup, salt, T_K, m, col))
            total = total + m_pair * mc12_pair_lambda(PAIR_LAMBDA_FOR_COL[col], T_K, I)
    return total

# ==========================================================
# SCORING
# ==========================================================
def _metrics(kappa, pred, T_K, m):
    kappa = np.asarray(kappa, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(pred) & np.isfinite(kappa) & (kappa > 0)
    if ok.sum() < 3:
        return None, ok
    k, p = kappa[ok], pred[ok]
    T, mm = np.asarray(T_K, float)[ok], np.asarray(m, float)[ok]
    resid = p - k
    dev = 100.0 * resid / k
    ss_tot = np.sum((k - k.mean()) ** 2)
    out = {
        'n': int(ok.sum()),
        'r2': float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        'rmse_abs': float(np.sqrt(np.mean(resid ** 2))),
        'MAE_pct': float(np.mean(np.abs(dev))),
        'RMSE_pct': float(np.sqrt(np.mean(dev ** 2))),
        'bias_pct': float(np.mean(dev)),
        'worst_pct': float(dev[np.argmax(np.abs(dev))]),
        'devT_slope': float(np.polyfit(T, dev, 1)[0]) if len(np.unique(T)) > 1 else np.nan,
        'r_dev_m': float(sstats.pearsonr(mm, dev)[0]) if len(np.unique(mm)) > 2 else np.nan,
    }
    return out, ok

def _print_line(name, r):
    if r is None:
        print(f"    {name:26s} not computable (coverage/coefficients).")
        return
    print(f"    {name:26s} n={r['n']:3d}  MAE={r['MAE_pct']:6.2f}%  "
          f"RMSE={r['RMSE_pct']:6.2f}%  bias={r['bias_pct']:+6.2f}%  "
          f"worst={r['worst_pct']:+7.2f}%  dev-T={r['devT_slope']:+.3f} %/K")

def _load_salt_test_rows(salt, sources=None):
    lib = pd.read_csv(_data(LIBRARY_FILE))
    d = lib[(lib['Salt'] == salt) &
            (lib['Pressure_MPa'] == REFERENCE_PRESSURE_MPA) &
            lib['Conductivity_mScm'].notna()].copy()
    d = d[~d['Notes'].astype(str).str.contains('FLAGGED', case=False, na=False)]
    if 'Source_Type' in d.columns:
        d = d[d['Source_Type'].astype(str).str.lower() != 'derived']
    if sources is not None:
        d = d[d['Source'].isin(sources)]
    if 'Mixture' in d.columns:
        d = d[d['Mixture'] != True]
    d = d.rename(columns={'Molality_molkg': 'Molality', 'Temperature_C': 'Temperature',
                          'Conductivity_mScm': 'Conductivity'})
    d['temp_K'] = d['Temperature'] + 273.15
    return d

ADDITIVITY_RESULTS = []
_STYLES = {}  # combo -> (color, marker, ls); filled at runtime

def additivity_test(mdl, salt, d, label, circular=False, extrap_mode=TEST_EXTRAP_MODE,
                    fig_tag=None, include_fw_alt=True):
    """Score up to four predictors on the same measured points."""
    d = d.copy()
    T_K = d['temp_K'].values
    m = d['Molality'].values
    kappa = d['Conductivity'].values
    tag = 'IN-SAMPLE CONSISTENCY (circular)' if circular else 'NON-CIRCULAR TEST'
    print("\n" + "-" * 72)
    print(f"{label}  [{tag}]")
    preds = {}
    preds[f'framework+{NATIVE_DB}'] = np.atleast_1d(
        mdl.predict_from_speciation(salt, T_K, m, extrap_mode=extrap_mode))
    if include_fw_alt and spec_lookup_alt is not None and _has_salt(spec_lookup_alt, salt):
        mdl_alt = make_model_with(spec_lookup_alt, base=mdl)
        try:
            preds[f'framework+{ALT_DB}*'] = np.atleast_1d(
                mdl_alt.predict_from_speciation(salt, T_K, m, extrap_mode=extrap_mode))
        except KeyError as e:
            print(f"    framework+{ALT_DB}*: skipped ({e})")
    for db, lk in ENGINES.items():
        preds[f'McCleskey+{db}'] = mccleskey_predict(salt, T_K, m, lk)
    # coverage anchored on the framework-native prediction
    base_key = f'framework+{NATIVE_DB}'
    cover = np.isfinite(preds[base_key])
    n_all = len(d)
    print(f"  coverage: {int(cover.sum())}/{n_all} points inside the framework's "
          f"spline hulls (extrap='{extrap_mode}'); all predictors scored on these points")
    if cover.sum() and cover.sum() < n_all:
        d_out = d[~cover]
        print(f"  uncovered rows: T {d_out['Temperature'].min():.1f} to "
              f"{d_out['Temperature'].max():.1f} C, m {d_out['Molality'].min():.4f} "
              f"to {d_out['Molality'].max():.4f} mol/kg")
    if cover.sum() == 0:
        print("  NO reconstructable points - skipping.")
        return None
    results = {}
    for name, p in preds.items():
        r, _ = _metrics(kappa, np.where(cover, p, np.nan), T_K, m)
        results[name] = r
        _print_line(name, r)
    # verdicts within each engine (fitting-strategy effect)
    for db in ENGINES:
        fw_key = base_key if db == NATIVE_DB else f'framework+{db}*'
        mc_key = f'McCleskey+{db}'
        if results.get(fw_key) and results.get(mc_key):
            diff = results[mc_key]['RMSE_pct'] - results[fw_key]['RMSE_pct']
            who = 'framework' if diff > 0 else 'McCleskey'
            print(f"    [{db}] fitting-strategy verdict: {who} better by "
                  f"{abs(diff):.2f} RMSE% points")
    # engine effect within each model
    if len(ENGINES) == 2:
        for mdl_name, keys in [('framework', (base_key, f'framework+{ALT_DB}*')),
                               ('McCleskey', (f'McCleskey+{NATIVE_DB}', f'McCleskey+{ALT_DB}'))]:
            a, b = results.get(keys[0]), results.get(keys[1])
            if a and b:
                print(f"    engine effect on {mdl_name}: bias {a['bias_pct']:+.2f}% "
                      f"({NATIVE_DB}) vs {b['bias_pct']:+.2f}% ({ALT_DB})")
    # per-source breakdown (framework native)
    if 'Source' in d.columns and d['Source'].nunique() > 1:
        dev_fw = 100.0 * (preds[base_key] - kappa) / kappa
        for src in sorted(d['Source'].unique()):
            s_mask = (d['Source'] == src).values & cover & np.isfinite(dev_fw)
            if s_mask.sum():
                print(f"      {src:15s}: n={int(s_mask.sum()):3d}  "
                      f"MAE={np.mean(np.abs(dev_fw[s_mask])):6.2f}%  "
                      f"bias={np.mean(dev_fw[s_mask]):+6.2f}%   (framework+{NATIVE_DB})")
    # ---- figure ----
    palette = [('#1f77b4', 'o', '-'), ('#17becf', 's', '--'),
               ('#ff7f0e', 'X', ':'), ('#d62728', '^', '-.')]
    for i, name in enumerate(preds):
        if name not in _STYLES:
            _STYLES[name] = palette[min(i, len(palette) - 1)]
    k_ok = kappa[cover]
    m_ok = m[cover]
    order = np.argsort(m_ok)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    lims = [k_ok.min() * 0.8, k_ok.max() * 1.2]
    ax1.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax2.axhspan(-5, 5, color='#e0e0e0', alpha=0.5, label=r'$\pm$5% window')
    for name, p in preds.items():
        if results.get(name) is None:
            continue
        c, mk, ls = _STYLES[name]
        p_ok = p[cover]
        ax1.loglog(k_ok, p_ok, mk, color=c, ms=4.5, mec='black', mew=0.3,
                   label=name, alpha=0.8)
        ax2.plot(m_ok[order], (100 * (p_ok - k_ok) / k_ok)[order], marker=mk, ls=ls,
                 color=c, lw=0.8, ms=4, label=name, alpha=0.85)
    ax1.set_xlabel(r'measured $\kappa$ (mS cm$^{-1}$)')
    ax1.set_ylabel(r'predicted $\kappa$ (mS cm$^{-1}$)')
    ax1.set_title(f'{salt} - parity')
    ax1.legend(loc='lower right', fontsize=7)
    ax1.grid(True, which='both', ls=':', alpha=0.3)
    ax2.axhline(0, color='k', lw=1)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'molality (mol kg$^{-1}$)')
    ax2.set_ylabel('signed deviation (%)')
    ax2.set_title(f'{tag.split()[0].lower()} deviation')
    ax2.legend(loc='best', fontsize=7)
    ax2.grid(True, which='both', ls=':', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, fig_tag or f'additivity_{salt.replace(":", "_")}')
    plt.show()
    ADDITIVITY_RESULTS.append({'salt': salt, 'label': label, 'circular': circular,
                               'results': results})
    return results

# ==========================================================
# TEST 1: K2SO4 (held out, non-circular)
# ==========================================================
additivity_test(model, 'K2SO4', salt_dfs['K2SO4'],
                'K2SO4 from K+ (KCl) + SO4 (Na2SO4) + KSO4- pair')

# ==========================================================
# TEST 2: CaSO4 (held out, non-circular; very dilute)
# ==========================================================
d_caso4 = _load_salt_test_rows('CaSO4', sources=['McCleskey2011'])
if len(d_caso4):
    additivity_test(model, 'CaSO4', d_caso4,
                    'CaSO4 from Ca2+ (CaCl2) + SO4 (Na2SO4)')
else:
    print("\nCaSO4: no ambient rows found - skipped.")

# ==========================================================
# TEST 3: MgSO4 (held out, non-circular; the known hard case)
# ==========================================================
additivity_test(model, 'MgSO4', salt_dfs['MgSO4'],
                'MgSO4 from Mg2+ (MgCl2) + SO4 (Na2SO4)')

# ==========================================================
# TEST 4: CaCl2 (IN-SAMPLE consistency - circular by construction)
# ==========================================================
additivity_test(model, 'CaCl2', salt_dfs['CaCl2'],
                'CaCl2 from Ca2+ + Cl- (self-consistency)', circular=True)

# ==========================================================
# TEST 5: Na2SO4 via CHAIN B (non-circular)
# ==========================================================
# Chain B SO4 is deconvolved with the NATIVE engine, so only the native
# framework cell is scored (an engine swap would be doubly inconsistent);
# both McCleskey engine cells still run.
print("\n" + "=" * 72)
print("CHAIN B: refitting SO4 2- from K2SO4 via K+ (for the Na2SO4 test)")
print("=" * 72)
model_b = IonConductivityModel(speciation=spec_lookup_native)
model_b.add_ion('K', spline_k)
model_b.add_ion('Cl', spline_cl)
model_b.add_ion('Na', spline_na)
spline_so4_b, so4b_t_range, so4b_I_range, df_so4b_ion = subtract_ion_molar(
    salt_dfs['K2SO4'].copy(), 'K2SO4', known_ion='K', model=model_b,
    new_ion_pretty_label='SO4 2- (chain B, from K2SO4)',
    known_ion_extrap_mode='linear', P_col='Pressure_MPa')

_tg = np.linspace(max(so4_t_range[0], so4b_t_range[0]),
                  min(so4_t_range[1], so4b_t_range[1]), 25)
_Ig = np.geomspace(max(so4_I_range[0], so4b_I_range[0], 1e-4),
                   min(so4_I_range[1], so4b_I_range[1]), 25)
_TT, _II = np.meshgrid(_tg, _Ig)
_a = spline_so4.ev_or_nan(_TT.ravel(), _II.ravel())
_b = spline_so4_b.ev_or_nan(_TT.ravel(), _II.ravel())
_bothok = np.isfinite(_a) & np.isfinite(_b) & (_a > 0)
if _bothok.any():
    _dd = 100.0 * (_b[_bothok] - _a[_bothok]) / _a[_bothok]
    print(f"SO4 chain A (Na2SO4) vs chain B (K2SO4) on the overlap grid: "
          f"n={_bothok.sum()}, mean diff={np.mean(_dd):+.2f}%, "
          f"mean |diff|={np.mean(np.abs(_dd)):.2f}%, worst={_dd[np.argmax(np.abs(_dd))]:+.2f}%")

additivity_test(model_b, 'Na2SO4', salt_dfs['Na2SO4'],
                'Na2SO4 from Na+ (NaCl) + SO4 (chain B: K2SO4) + NaSO4- pair',
                fig_tag='additivity_Na2SO4_chainB', include_fw_alt=False)

# ==========================================================
# SUMMARY GRID (MAE% | bias%, same covered points per test)
# ==========================================================
print("\n" + "=" * 72)
print("2x2 SUMMARY - MAE% (bias%) per predictor, same points within each test")
print("=" * 72)
_all_keys = []
for res in ADDITIVITY_RESULTS:
    for k in res['results']:
        if k not in _all_keys:
            _all_keys.append(k)
hdr = f"{'test':18s}" + ''.join(f"{k:>26s}" for k in _all_keys)
print(hdr)
for res in ADDITIVITY_RESULTS:
    name = res['salt'] + ('(circ)' if res['circular'] else '')
    row = f"{name:18s}"
    for k in _all_keys:
        r = res['results'].get(k)
        row += f"{'-':>26s}" if r is None else \
            f"{r['MAE_pct']:>13.2f} ({r['bias_pct']:+7.2f})"
    print(row)
print(f"\n* = prediction-time engine swap; splines trained with {NATIVE_DB}. "
      f"For the consistent {ALT_DB} framework, rerun Cell 1 with "
      f"SPECIATION_DB='{ALT_DB}' and run these tests again.")
print("Attribution: framework-vs-McCleskey within one engine isolates the "
      "FITTING STRATEGY; supcrt-vs-wateq4f within one model isolates the "
      "SPECIATION ENGINE. A shared miss under all four = shared physics "
      "(e.g. pair conduction). CaCl2 is self-consistency only.")
