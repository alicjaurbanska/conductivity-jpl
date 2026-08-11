# ==========================================================
# SUFFICIENCY DIAGNOSTICS - noise floor + leave-one-dataset-out
# ==========================================================
#
# the code answers 2 questions:
#   1. NOISE FLOOR: is the training error already at the reproducibility
#      limit of the data? Compares (a) within-source replicate spread,
#      (b) between-source closure offsets, (c) model training RMSE.
#      If (c) <= (a),(b), the surfaces cannot be legitimately tightened.
#   2. LODO: rebuild the ENTIRE sequential chain with one source excluded,
#      then evaluate on that source's rows - the true out-of-sample
#      accuracy of the framework, replacing the in-sample caveat with
#      numbers. McCleskey2011 cannot be excluded (sole transport-number
#      anchor); that non-excludability is itself a documented limitation.
#

import io
import contextlib
import numpy as np
import pandas as pd

assert FIT_MODE == 'sequential', 'assumes the sequential production fit in memory'

LODO_SOURCES = ['Cortes2026', 'Mahboub']   # extend if more sources are added
QUIET_LODO = True

def _q():
    return (contextlib.redirect_stdout(io.StringIO()) if QUIET_LODO
            else contextlib.nullcontext())

# ==========================================
# 1. NOISE FLOOR
# ==========================================
print('=' * 70)
print('1. NOISE FLOOR')
print('=' * 70)

# (a) within-source repeatability: relative std of replicate groups
grp = df.groupby(['Salt', 'Source', '_tkey', '_mkey', '_pkey'])['Conductivity_mScm']
rep = grp.agg(['mean', 'std', 'count'])
rep = rep[rep['count'] > 1].copy()
if len(rep):
    rep['rel_std'] = rep['std'] / rep['mean']
    rep_by_salt = rep.groupby('Salt')['rel_std'].agg(['median', 'max', 'count'])
    print('\n  (a) Within-source replicate spread (relative std where n>1):')
    for salt, r in rep_by_salt.iterrows():
        print(f"      {salt:10s}: median {r['median']:.2%}, "
              f"max {r['max']:.2%} ({int(r['count'])} replicated conditions)")
    floor_within = float(rep['rel_std'].median())
else:
    print('\n  (a) No replicated conditions in the library.')
    floor_within = np.nan

# (b) between-source closure offsets per training salt
print('\n  (b) Between-source offsets (per-source mean relative deviation of '
      'the production model):')
noise_rows = []
for salt in FIT_SALTS:
    d = salt_dfs[salt]
    pred = np.atleast_1d(model.predict_from_speciation(
        salt, d['temp_K'].values, d['Molality'].values, extrap_mode='linear'))
    meas = d['Conductivity'].values
    ok = np.isfinite(pred) & (meas != 0)
    rel = (pred[ok] - meas[ok]) / meas[ok]
    train_rmse = float(np.sqrt(np.mean(rel ** 2)))
    src_means = pd.Series(rel).groupby(d['Source'].values[ok]).mean()
    spread = float(src_means.max() - src_means.min()) if len(src_means) > 1 else np.nan
    src_str = ', '.join(f'{s}: {v:+.1%}' for s, v in src_means.items())
    print(f"      {salt:8s}: train RMSE {train_rmse:.2%}; offsets [{src_str}]"
          + (f'; cross-source spread {spread:.1%}' if np.isfinite(spread) else ''))
    noise_rows.append(dict(salt=salt, train_rel_rmse=train_rmse,
                           cross_source_spread=spread,
                           n_sources=len(src_means)))
noise = pd.DataFrame(noise_rows)
noise.to_csv('noise_floor.csv', index=False)

print('\n  VERDICT (a)-(c): the surfaces are at the noise floor wherever '
      'train RMSE is at or below the replicate spread and well below the '
      'cross-source spread. Where cross-source spread dominates (e.g. an '
      '~11-14% source offset), tighter smoothing fits one laboratory\'s '
      'systematics, not physics - the fix is source reconciliation or '
      'exclusion, never more knots.')

# ==========================================
# 2. LEAVE-ONE-DATASET-OUT (full chain rebuild)
# ==========================================
def build_chain(exclude_source):
    """Rebuild the sequential chain exactly as in the main cell, with every
    row from `exclude_source` removed from every fit pool."""
    def filt(salt):
        d = salt_dfs[salt]
        return d[d['Source'] != exclude_source].copy()
    dk = filt('KCl')
    if dk.empty:
        raise RuntimeError('KCl pool empty after exclusion.')
    tp = calculate_t_kcl(dk['Temperature'].values, dk['Molality'].values)
    lam_salt = dk['Conductivity'].values / dk['m_free_K'].values
    lam_K = tp * lam_salt
    lam_Cl = (dk['Conductivity'].values / dk['m_free_Cl'].values) - \
        lam_K * dk['m_free_K'].values / dk['m_free_Cl'].values
    def _ion_df(lam):
        return pd.DataFrame({'Source': dk['Source'].values,
                             'temp_K': dk['temp_K'].values,
                             'I': dk['I'].values,
                             'Molality': dk['Molality'].values,
                             'Lambda': lam,
                             'Pressure_MPa': dk['Pressure_MPa'].values,
                             'P_state_MPa': dk['P_state_MPa'].values,
                             'Ambient': True,
                             'From_Extrapolated_Known': False})
    m2 = IonConductivityModel(speciation=spec_lookup)
    sK, *_ = fit_lambda_spline(_ion_df(lam_K), f'K [LODO -{exclude_source}]')
    sCl, *_ = fit_lambda_spline(_ion_df(lam_Cl), f'Cl [LODO -{exclude_source}]')
    m2.add_ion('K', sK)
    m2.add_ion('Cl', sCl)
    for salt, known, ion in [('NaCl', 'Cl', 'Na'), ('MgCl2', 'Cl', 'Mg'),
                             ('Na2SO4', 'Na', 'SO4'), ('CaCl2', 'Cl', 'Ca'),
                             ('NH4Cl', 'Cl', 'NH4')]:
        subtract_ion_molar(filt(salt), salt, known_ion=known, model=m2,
                           new_ion_pretty_label=f'{ion} [LODO -{exclude_source}]',
                           known_ion_extrap_mode='linear')
    return m2

print('\n' + '=' * 70)
print('2. LEAVE-ONE-DATASET-OUT (true out-of-sample accuracy)')
print('=' * 70)
print('  NOTE: McCleskey2011 is not excludable - it is the sole '
      'transport-number-anchored entry point of the chain. This dependence '
      'is a documented limitation and one more argument for new anchor-grade '
      'measurements.')
lodo_rows = []
for src in LODO_SOURCES:
    print(f"\n  Excluding {src} from every fit pool and rebuilding the chain...")
    try:
        with _q():
            m2 = build_chain(src)
    except Exception as e:
        print(f"    chain rebuild FAILED without {src}: {e}")
        continue
    any_rows = False
    for salt in FIT_SALTS:
        d = salt_dfs[salt]
        held = d[d['Source'] == src]
        if held.empty:
            continue
        any_rows = True
        T = held['temp_K'].values
        M = held['Molality'].values
        meas = held['Conductivity'].values
        with _q():
            pred = np.atleast_1d(m2.predict_from_speciation(
                salt, T, M, extrap_mode='linear'))
            pred0 = np.atleast_1d(model.predict_from_speciation(
                salt, T, M, extrap_mode='linear'))
        ok = np.isfinite(pred) & np.isfinite(pred0) & (meas != 0)
        n_oob = int((~np.isfinite(pred)).sum())
        if not ok.any():
            print(f"    {salt:8s}/{src}: all {len(held)} held-out rows "
                  f"outside the LODO hull - no score.")
            continue
        rel = (pred[ok] - meas[ok]) / meas[ok]
        rel0 = (pred0[ok] - meas[ok]) / meas[ok]
        lodo_rmse = float(np.sqrt(np.mean(rel ** 2)))
        lodo_bias = float(np.mean(rel))
        ins_rmse = float(np.sqrt(np.mean(rel0 ** 2)))
        ins_bias = float(np.mean(rel0))
        print(f"    {salt:8s}/{src}: LODO RMSE {lodo_rmse:.2%} "
              f"(bias {lodo_bias:+.2%}) vs in-sample {ins_rmse:.2%} "
              f"(bias {ins_bias:+.2%}) on {int(ok.sum())}/{len(held)} rows"
              + (f'; {n_oob} outside LODO hull' if n_oob else ''))
        lodo_rows.append(dict(salt=salt, excluded_source=src,
                              n_scored=int(ok.sum()), n_held=len(held),
                              n_out_of_hull=n_oob,
                              lodo_rel_rmse=lodo_rmse, lodo_bias=lodo_bias,
                              insample_rel_rmse=ins_rmse,
                              insample_bias=ins_bias))
    if not any_rows:
        print(f"    {src}: no rows in any FIT salt - nothing to score.")

if lodo_rows:
    pd.DataFrame(lodo_rows).to_csv('lodo_results.csv', index=False)
    print('\n  wrote lodo_results.csv')
    print('\n  READING GUIDE:')
    print('  - LODO RMSE close to in-sample RMSE -> the in-sample advantage '
          'is real; report LODO as the framework\'s out-of-sample accuracy '
          'and cite it next to the MC12 comparison.')
    print('  - LODO RMSE >> in-sample RMSE for one source only (expect '
          'Cortes2026, given its +11-14% offsets) -> the gap measures '
          'between-source disagreement, not model failure; it feeds the '
          'data-consistency discussion, not the smoothing choice.')
    print('  - The honest MC12 sentence follows: both frameworks are '
          'in-sample on their own training salts (K2SO4 and MgSO4 are in '
          'MC12\'s fit set); the informative comparisons are sub-zero '
          'temperatures and mixtures, plus LODO for this framework.')
