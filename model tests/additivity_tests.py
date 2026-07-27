# ==========================================================
# ADDITIVITY TESTS - reconstruct salt conductivities from ion splines
# ==========================================================
# Run AFTER Cell 1 (v5.2+) in the same notebook: this cell uses Cell 1's
# namespace (model, splines, salt_dfs, spec_lookup, subtract_ion_molar,
# IonConductivityModel, save_fig, _data, LIBRARY_FILE, ...).
#
# Colab usage (pull from GitHub):
#   import requests
#   exec(requests.get('https://raw.githubusercontent.com/alicjaurbanska/'
#        'conductivity-jpl/main/model%20tests/additivity_tests.py').text)
#
# TESTS (all ambient):
#   1. K2SO4  NON-CIRCULAR, held out: K+ (from KCl) + SO4 (from Na2SO4)
#             + KSO4- pair conduction. K2SO4 data never entered any fit.
#   2. CaSO4  NON-CIRCULAR, held out: Ca2+ (from CaCl2) + SO4 (from Na2SO4).
#   3. MgSO4  NON-CIRCULAR, held out: Mg2+ (from MgCl2) + SO4 (from Na2SO4);
#             pools: Mahboub + Horne + Larionov (ambient). Neutral MgSO4(aq)
#             pair conducts nothing - physics, not neglect.
#   4. CaCl2  IN-SAMPLE CONSISTENCY (circular): Ca2+ was deconvolved from
#             this very data; quantifies deconvolution+refit self-consistency,
#             NOT predictive skill. Labeled as such everywhere.
#   5. Na2SO4 NON-CIRCULAR via CHAIN B: SO4 is REFIT from K2SO4 via K+
#             (instead of from Na2SO4 via Na+), then Na2SO4 is reconstructed
#             from Na+ (from NaCl) + SO4_chainB + NaSO4- pair. The Na2SO4
#             data never touched either ingredient.
#
# Every test reports: coverage (points inside spline hulls), MAE%, RMSE%,
# mean bias%, worst deviation, deviation-vs-T slope, per-source breakdown,
# and saves a two-panel figure (parity + percent deviation vs T, colored by
# molality) - deviation plots per Steve's standing expectation.

print("=" * 70)
print("ADDITIVITY TESTS (ambient reconstruction from ion splines)")
print("=" * 70)

TEST_EXTRAP_MODE = 'linear'   # Walden T-extension for rows below spline T floors
                              # (e.g. Mahboub -10 C, Horne -2.3 C); set None to
                              # restrict tests to strict hull coverage.

def _load_salt_test_rows(salt, sources=None):
    """Ambient absolute-conductivity rows for a salt straight from the
    library (for salts Cell 1 does not stage, e.g. CaSO4). Applies the same
    hygiene: FLAGGED out, derived out, relative-only rows out."""
    lib = pd.read_csv(_data(LIBRARY_FILE))
    d = lib[(lib['Salt'] == salt) &
            (lib['Pressure_MPa'] == REFERENCE_PRESSURE_MPA) &
            lib['Conductivity_mScm'].notna()].copy()
    flagged = d['Notes'].astype(str).str.contains('FLAGGED', case=False, na=False)
    d = d[~flagged]
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

def additivity_test(mdl, salt, d, label, circular=False, extrap_mode=TEST_EXTRAP_MODE,
                    fig_tag=None):
    """Reconstruct kappa(salt) at the data's (T, m) from ion splines via
    speciation, compare to measurement, print metrics, save deviation figure.
    Returns a metrics dict (also appended to ADDITIVITY_RESULTS)."""
    d = d.copy()
    T_K = d['temp_K'].values
    m = d['Molality'].values
    kappa = d['Conductivity'].values
    pred = np.atleast_1d(mdl.predict_from_speciation(salt, T_K, m,
                                                     extrap_mode=extrap_mode))
    ok = np.isfinite(pred) & np.isfinite(kappa) & (kappa > 0)
    n_all, n_ok = len(d), int(ok.sum())
    print("\n" + "-" * 70)
    tag = 'IN-SAMPLE CONSISTENCY (circular)' if circular else 'NON-CIRCULAR TEST'
    print(f"{label}  [{tag}]")
    print(f"  coverage: {n_ok}/{n_all} points reconstructable "
          f"(rest outside ion-spline hulls even with extrap='{extrap_mode}')")
    if n_ok and n_ok < n_all:
        d_out = d[~ok]
        print(f"  uncovered rows: T {d_out['Temperature'].min():.1f} to "
              f"{d_out['Temperature'].max():.1f} C, "
              f"m {d_out['Molality'].min():.4f} to {d_out['Molality'].max():.4f} mol/kg")
    if n_ok == 0:
        print("  NO reconstructable points - skipping metrics.")
        return None
    dev = 100.0 * (pred[ok] - kappa[ok]) / kappa[ok]
    Tt = T_K[ok]
    slope = np.polyfit(Tt, dev, 1)[0] if len(np.unique(Tt)) > 1 else np.nan
    res = {
        'salt': salt, 'label': label, 'circular': circular,
        'n': n_ok, 'n_total': n_all,
        'MAE_pct': float(np.mean(np.abs(dev))),
        'RMSE_pct': float(np.sqrt(np.mean(dev ** 2))),
        'bias_pct': float(np.mean(dev)),
        'worst_pct': float(dev[np.argmax(np.abs(dev))]),
        'dev_T_slope_pct_per_K': float(slope),
    }
    print(f"  MAE={res['MAE_pct']:.2f}%  RMSE={res['RMSE_pct']:.2f}%  "
          f"bias={res['bias_pct']:+.2f}%  worst={res['worst_pct']:+.2f}%  "
          f"dev-vs-T slope={res['dev_T_slope_pct_per_K']:+.3f} %/K")
    if 'Source' in d.columns and d['Source'].nunique() > 1:
        for src, g_idx in d[ok].groupby('Source').groups.items():
            sel = d.index.get_indexer(g_idx)
            src_mask = ok.copy()
            src_mask[:] = False
            src_mask[sel] = True
            src_dev = 100.0 * (pred[src_mask] - kappa[src_mask]) / kappa[src_mask]
            print(f"    {src:15s}: n={len(src_dev):3d}  MAE={np.mean(np.abs(src_dev)):.2f}%  "
                  f"bias={np.mean(src_dev):+.2f}%")
    # ---- deviation figure (parity + dev vs T colored by molality) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(kappa[ok], pred[ok], 'o', ms=4, alpha=0.7)
    lims = [min(kappa[ok].min(), pred[ok].min()) * 0.8,
            max(kappa[ok].max(), pred[ok].max()) * 1.2]
    ax1.plot(lims, lims, 'k--', lw=1)
    ax1.set_xlabel(r'measured $\kappa$ (mS cm$^{-1}$)')
    ax1.set_ylabel(r'reconstructed $\kappa$ (mS cm$^{-1}$)')
    ax1.set_title(f'{label} - parity')
    sc = ax2.scatter(Tt - 273.15, dev, c=m[ok], cmap='viridis', s=18)
    ax2.axhline(0, color='k', lw=1)
    ax2.set_xlabel('T (C)')
    ax2.set_ylabel('deviation (%)')
    ax2.set_title(f'{tag.split()[0].lower()} deviation vs T')
    fig.colorbar(sc, ax=ax2, label='molality (mol/kg)')
    fig.tight_layout()
    save_fig(fig, fig_tag or f'additivity_{salt.replace(":", "_")}')
    plt.show()
    ADDITIVITY_RESULTS.append(res)
    return res

ADDITIVITY_RESULTS = []

# ==========================================
# TEST 1: K2SO4 (held out, non-circular)
# ==========================================
additivity_test(model, 'K2SO4', salt_dfs['K2SO4'],
                'K2SO4 from K+ (KCl) + SO4 (Na2SO4) + KSO4- pair')

# ==========================================
# TEST 2: CaSO4 (held out, non-circular; very dilute, m <= 0.012)
# ==========================================
d_caso4 = _load_salt_test_rows('CaSO4', sources=['McCleskey2011'])
if len(d_caso4):
    additivity_test(model, 'CaSO4', d_caso4,
                    'CaSO4 from Ca2+ (CaCl2) + SO4 (Na2SO4)')
else:
    print("\nCaSO4: no ambient rows found - skipped.")

# ==========================================
# TEST 3: MgSO4 (held out, non-circular; the known hard case)
# ==========================================
additivity_test(model, 'MgSO4', salt_dfs['MgSO4'],
                'MgSO4 from Mg2+ (MgCl2) + SO4 (Na2SO4)')

# ==========================================
# TEST 4: CaCl2 (IN-SAMPLE consistency - circular by construction)
# ==========================================
additivity_test(model, 'CaCl2', salt_dfs['CaCl2'],
                'CaCl2 from Ca2+ + Cl- (self-consistency)', circular=True)

# ==========================================
# TEST 5: Na2SO4 via CHAIN B (non-circular)
# ==========================================
# Chain B: SO4 refit from K2SO4 data via the K+ spline (KSO4- pair
# subtracted inside subtract_ion_molar), in a SEPARATE model object so the
# main model's SO4 (chain A, from Na2SO4) is untouched. Then Na2SO4 is
# reconstructed from Na+ (from NaCl) + SO4_chainB + NaSO4- pair.
print("\n" + "=" * 70)
print("CHAIN B: refitting SO4 2- from K2SO4 via K+ (for the Na2SO4 test)")
print("=" * 70)
model_b = IonConductivityModel(speciation=spec_lookup)
model_b.add_ion('K', spline_k)
model_b.add_ion('Cl', spline_cl)
model_b.add_ion('Na', spline_na)
spline_so4_b, so4b_t_range, so4b_I_range, df_so4b_ion = subtract_ion_molar(
    salt_dfs['K2SO4'].copy(), 'K2SO4', known_ion='K', model=model_b,
    new_ion_pretty_label='SO4 2- (chain B, from K2SO4)',
    known_ion_extrap_mode='linear', P_col='Pressure_MPa')

# How different are the two independently derived SO4 surfaces?
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
                fig_tag='additivity_Na2SO4_chainB')

# ==========================================
# SUMMARY TABLE
# ==========================================
print("\n" + "=" * 70)
print("ADDITIVITY TEST SUMMARY")
print("=" * 70)
print(f"{'test':55s} {'n':>5s} {'MAE%':>7s} {'RMSE%':>7s} {'bias%':>8s} {'%/K':>8s}")
for r in ADDITIVITY_RESULTS:
    flag = ' (circular)' if r['circular'] else ''
    print(f"{(r['label'][:52] + flag):55s} {r['n']:5d} {r['MAE_pct']:7.2f} "
          f"{r['RMSE_pct']:7.2f} {r['bias_pct']:+8.2f} "
          f"{r['dev_T_slope_pct_per_K']:+8.3f}")
print("\nReading guide: non-circular rows are the honest numbers. A flat "
      "deviation-vs-T slope with a nonzero bias suggests a multiplicative "
      "(speciation log K or calibration) offset; a strong slope suggests a "
      "mobility/temperature-channel error. CaCl2 is self-consistency only.")
