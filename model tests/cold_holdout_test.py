# ==========================================================
# COLD-HOLDOUT TEST: FIVE-MODEL COMPARISON (ambient rows only)
# Run AFTER the model cell and AFTER the McCleskey cell.
#
# Question: how well does the chain predict subzero conductivity
#   (a) when it has never seen subzero data (extrapolation), and
#   (b) in the production setup, which trains through the cold rows?
#
# Models:
#   mc     McCleskey et al. (2012), fed the same speciation as the chain
#   h_raw  chain fit on T >= 0 C rows only, raw lambda (alpha = 0)
#   h_wal  chain fit on T >= 0 C rows only, Walden (fitted alpha)
#   f_raw  chain fit on all ambient rows, raw lambda
#   f_wal  chain fit on all ambient rows, Walden (= production setup)
#
# Scored rows: every ambient T < 0 C row of every salt in salt_dfs.
#   fit salts:      full chains trained on these rows -> IN-SAMPLE
#   held-out salts: no chain trained on them -> OUT-OF-SAMPLE for all
# The common domain is fixed by the HOLDOUT chains and the MC12 I cap
# only, so it does not depend on the test rows.
#
# Raw variants reuse the production fitting code unchanged: IonSpline is
# temporarily swapped for a subclass with alpha fixed at 0, so
# W = lambda * eta^0 = lambda. The production `model` is not touched.
# All predictions go through IonConductivityModel.predict_conductivity
# with extrap_mode='linear', exactly as in production.
# ==========================================================
T_HOLD_K = 273.15
I_MAX_MC12 = 1.0          # MC12 ionic-strength cap for the common domain
N_BOOT = 2000
MIN_CLUSTERS_CI = 6       # below this the bootstrap CI is flagged unreliable
CHAIN_STEPS = [('NaCl', 'Cl'), ('MgCl2', 'Cl'), ('Na2SO4', 'Na'),
               ('CaCl2', 'Cl'), ('NH4Cl', 'Cl')]   # same order as production
PAIR_KEY_FOR_COL = {'m_pair_KSO4': 'KSO4', 'm_pair_NaSO4': 'NaSO4',
                    'm_pair_MgSO4': 'MgSO4'}
VKEYS = ['h_raw', 'h_wal', 'f_raw', 'f_wal']
MODELS = ['mc'] + VKEYS
NAME = {'mc': 'MC12', 'h_raw': 'Hold raw', 'h_wal': 'Hold Walden',
        'f_raw': 'Full raw', 'f_wal': 'Full Walden'}
ROLE_LABEL = {'fit': 'fit salts (full chains IN-SAMPLE)',
              'heldout': 'held-out salts (all models OUT-OF-SAMPLE)'}

# ---------- raw-lambda spline (alpha = 0) ----------
if getattr(IonSpline, '_is_raw_variant', False):   # left over from an interrupted run
    IonSpline = IonSpline.__mro__[1]
_IONSPLINE_ORIG = IonSpline

class RawLambdaSpline(_IONSPLINE_ORIG):
    """IonSpline with the Walden exponent fixed at 0 (fits lambda itself)."""
    _is_raw_variant = True
    @staticmethod
    def _fit_alpha(t, I, lam):
        return 0.0

# ---------- split data ----------
held, warm_fit, full_fit = {}, {}, {}
for salt, d in salt_dfs.items():
    amb = d[d['Ambient']]
    cold = amb['temp_K'].values < T_HOLD_K
    if salt in FIT_SALTS:
        warm_fit[salt] = amb[~cold].copy()
        full_fit[salt] = amb.copy()
    if cold.any():
        held[salt] = amb[cold].copy()
        role = 'fit salt' if salt in FIT_SALTS else 'held-out salt'
        print(f"{salt:7s} [{role}]: {int(cold.sum())} ambient subzero rows "
              f"({amb['temp_K'].values[cold].min() - 273.15:.1f} to "
              f"{amb['temp_K'].values[cold].max() - 273.15:.1f} C)")
if not held:
    raise RuntimeError('No ambient subzero rows found in any salt.')

# ---------- fit the four chains ----------
def fit_chain(fit_data, walden, tag):
    global IonSpline
    IonSpline = _IONSPLINE_ORIG if walden else RawLambdaSpline
    try:
        print(f"\n--- fitting chain [{tag}] ---")
        mdl = IonConductivityModel(speciation=spec_lookup)
        d = fit_data['KCl'].copy()
        tp = calculate_t_kcl(d['Temperature'].values, d['Molality'].values)
        lam_K = tp * d['Conductivity'] / d['m_free_K']
        lam_Cl = d['Conductivity'] / d['m_free_Cl'] - \
            lam_K * d['m_free_K'] / d['m_free_Cl']
        for ion, lam in [('K', lam_K), ('Cl', lam_Cl)]:
            di = pd.DataFrame({'Source': d['Source'], 'temp_K': d['temp_K'],
                               'I': d['I'], 'Molality': d['Molality'],
                               'Lambda': lam, 'From_Extrapolated_Known': False})
            spl, *_ = fit_lambda_spline(di, f'{ion} (from KCl) [{tag}]',
                                        anchor_ion=ion)
            mdl.add_ion(ion, spl)
        for s, known in CHAIN_STEPS:
            subtract_ion_molar(fit_data[s].copy(), s, known_ion=known, model=mdl,
                               new_ion_pretty_label=f'{s} new ion [{tag}]',
                               known_ion_extrap_mode='linear')
        return mdl
    finally:
        IonSpline = _IONSPLINE_ORIG

mdls = {
    'h_raw': fit_chain(warm_fit, False, 'holdout raw'),
    'h_wal': fit_chain(warm_fit, True,  'holdout Walden'),
    'f_raw': fit_chain(full_fit, False, 'full raw'),
    'f_wal': fit_chain(full_fit, True,  'full Walden'),
}

# ---------- prediction helpers ----------
def _frame_comp(fr):
    free = {ion: np.nan_to_num(fr[col].to_numpy(float))
            for ion, col in ION_FREE_COL.items() if col in fr.columns}
    pairs = {key: np.nan_to_num(fr[col].to_numpy(float))
             for col, key in PAIR_KEY_FOR_COL.items() if col in fr.columns}
    return {'I_eff': fr['I'].to_numpy(float), 'free_molality': free,
            'pair_molality': pairs, 'speciation_source': SPECIATION_DB}

def chain_kappa(mdl, fr, T_K):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.atleast_1d(mdl.predict_conductivity(
            _frame_comp(fr), np.asarray(T_K, float), extrap_mode='linear'))

def mc12_kappa(fr, T_K):
    species = {}
    for col in list(ION_FREE_COL.values()) + PAIR_COLS:
        if col in fr.columns:
            v = fr[col].to_numpy(float)
            if np.any(np.isfinite(v)):
                species[col] = np.nan_to_num(v)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = mccleskey2012_conductivity(species, np.asarray(T_K, float) - 273.15,
                                         I_eff_molkg=fr['I'].to_numpy(float),
                                         validate_range=False)
    return np.atleast_1d(np.asarray(res.sigma_mScm, dtype=float))

def holdout_evaluable(salt, T_K, I):
    """True where BOTH holdout chains can evaluate every ion of the salt
    (in hull, or via the production cold-side tangent extrapolation)."""
    ok = np.ones(len(T_K), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for key in ('h_raw', 'h_wal'):
            mdl = mdls[key]
            for ion in mdl.stoich_for(salt):
                spl = mdl.ion(ion)
                ok &= np.isfinite(spl.ev_or_nan(T_K, I)) | \
                    np.isfinite(spl.ev_extrap(T_K, I, mode='linear'))
    return ok

def chain_floor_C(salt):
    """Holdout-chain training floor: extrapolation starts below this."""
    return max(mdls[k].ion(ion).t_range[0] for k in ('h_raw', 'h_wal')
               for ion in mdls[k].stoich_for(salt)) - 273.15

def _rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

# ---------- sanity: full Walden chain must reproduce production ----------
_dev = []
for salt in FIT_SALTS:
    d = full_fit[salt]
    a = chain_kappa(mdls['f_wal'], d, d['temp_K'].values)
    b = chain_kappa(model, d, d['temp_K'].values)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.any():
        _dev.append(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok])))
_maxdev = max(_dev) if _dev else np.nan
print(f"\nReproducibility check: full Walden chain vs production model, "
      f"max rel. difference on training rows = {_maxdev:.2e}"
      + ("" if _maxdev < 1e-6 else "  <-- WARNING: should be ~0, check setup"))

print("\nWalden exponents alpha_i (raw chains have alpha = 0):")
for ion in SEQ_SPLINES:
    print(f"  {ion:4s}: holdout {mdls['h_wal'].ion(ion).alpha:.3f}   "
          f"full {mdls['f_wal'].ion(ion).alpha:.3f}   "
          f"production {SEQ_SPLINES[ion].alpha:.3f}")

# ---------- score the cold rows ----------
recs, excluded = [], []
for salt, h in held.items():
    T_K = h['temp_K'].to_numpy(float)
    I = h['I'].to_numpy(float)
    meas = h['Conductivity'].to_numpy(float)
    ok_chain = holdout_evaluable(salt, T_K, I) & (meas > 0)
    ok_mc = I <= I_MAX_MC12
    for reason, m in [('I > MC12 cap only', ok_chain & ~ok_mc),
                      ('holdout chain cannot evaluate only', ~ok_chain & ok_mc),
                      ('outside both', ~ok_chain & ~ok_mc)]:
        if m.any():
            excluded.append(f"  {salt:8s} {int(m.sum()):3d} rows: {reason} "
                            f"(I {I[m].min():.2f}-{I[m].max():.2f}, "
                            f"T {T_K[m].min() - 273.15:.1f} to "
                            f"{T_K[m].max() - 273.15:.1f} C)")
    if not ok_chain.any():
        print(f"{salt}: no cold rows inside the holdout-chain domain.")
        continue
    hs = h[ok_chain]
    Tk = T_K[ok_chain]
    preds = {k: chain_kappa(mdls[k], hs, Tk) for k in VKEYS}
    for k in ('h_raw', 'h_wal'):
        if not np.all(np.isfinite(preds[k])):
            raise RuntimeError(f'{salt}: {k} non-finite inside its own domain '
                               f'- domain definition is wrong.')
    full_ok = np.isfinite(preds['f_raw']) & np.isfinite(preds['f_wal'])
    if (~full_ok).any():
        print(f"{salt}: {int((~full_ok).sum())} row(s) in the holdout domain "
              f"that a FULL chain cannot evaluate - dropped from scoring.")
    sub = ok_mc[ok_chain]
    p_mc = np.full(len(hs), np.nan)
    if sub.any():
        p_mc[sub] = mc12_kappa(hs[sub], Tk[sub])
        if not np.all(np.isfinite(p_mc[sub])):
            raise RuntimeError(f'{salt}: MC12 non-finite inside its domain.')
    df_r = pd.DataFrame({
        'salt': salt, 'role': 'fit' if salt in FIT_SALTS else 'heldout',
        'source': hs['Source'].astype(str).values,
        'T_C': Tk - 273.15, 'depth_C': chain_floor_C(salt) - (Tk - 273.15),
        'meas': meas[ok_chain], 'p_mc': p_mc,
        'common': sub & full_ok, 'chain_ok': full_ok})
    for k in VKEYS:
        df_r[f'p_{k}'] = preds[k]
    recs.append(df_r)
if not recs:
    raise RuntimeError('No scorable cold rows.')
rows = pd.concat(recs, ignore_index=True)
for k in MODELS:
    rows[f'e_{k}'] = (rows[f'p_{k}'] - rows['meas']) / rows['meas']
rows['cluster'] = rows['salt'] + '|' + rows['source']
com = rows[rows['common']]
chain_rows = rows[rows['chain_ok']]

# ---------- tables ----------
def _cell(g, k):
    e = g[f'e_{k}']
    return f"{_rmse(e):.1%}/{100 * e.mean():+.1f}%"

def print_table(df, models, title):
    print('\n' + '=' * 96)
    print(title)
    print(f"{'role':8s} {'salt':8s} {'n':>4s} "
          + ''.join(f"{NAME[k]:>16s}" for k in models)
          + '   (rel. RMSE / mean bias)')
    for role, gr in df.groupby('role'):
        for salt, g in gr.groupby('salt'):
            print(f"{role:8s} {salt:8s} {len(g):4d} "
                  + ''.join(f"{_cell(g, k):>16s}" for k in models))
        print(f"{role:8s} {'POOLED':8s} {len(gr):4d} "
              + ''.join(f"{_cell(gr, k):>16s}" for k in models))
        eq = {k: np.mean([_rmse(g[f'e_{k}']) for _, g in gr.groupby('salt')])
              for k in models}
        print(f"{role:8s} {'EQ-WT':8s} {gr['salt'].nunique():4d} "
              + ''.join(f"{eq[k]:>16.1%}" for k in models))
    print("  EQ-WT = mean of per-salt RMSEs. Holdout chains EXTRAPOLATE below "
          "their floor; full chains on fit salts are IN-SAMPLE.")

print_table(com, MODELS, 'COMMON DOMAIN (identical rows, all five models, '
            f'I <= {I_MAX_MC12:g} mol/kg)')
print_table(chain_rows, VKEYS, 'CHAIN-ONLY DOMAIN (no MC12 I cap; includes '
            'concentrated rows)')

print("\nRows excluded from scoring (reported, not scored):")
for line in excluded or ['  none']:
    print(line)

# ---------- paired cluster bootstrap ----------
def boot_rmse_diff(df, col_a, col_b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    clusters = df['cluster'].unique()
    if len(clusters) < 2:
        return None
    groups = {c: df[df['cluster'] == c] for c in clusters}
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        s = pd.concat([groups[c] for c in pick])
        diffs[b] = _rmse(s[col_a]) - _rmse(s[col_b])
    point = _rmse(df[col_a]) - _rmse(df[col_b])
    return (point, float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), len(clusters))

COMPARISONS = [
    ('h_wal', 'h_raw', 'Hold Walden - Hold raw    (does Walden extrapolate better?)'),
    ('h_wal', 'mc',    'Hold Walden - MC12        (extrapolation vs MC12)'),
    ('f_wal', 'mc',    'Full Walden - MC12        (production vs MC12)'),
    ('f_wal', 'h_wal', 'Full Walden - Hold Walden (value of cold training data)'),
]
print(f"\nPaired cluster bootstrap on the common domain ({N_BOOT} draws, "
      "clusters = salt x source), 95% CI on RMSE difference; "
      "negative -> first model better:")
for role, gr in com.groupby('role'):
    print(f"  [{ROLE_LABEL[role]}]")
    for a, b, label in COMPARISONS:
        out = boot_rmse_diff(gr, f'e_{a}', f'e_{b}')
        if out is None:
            print(f"    {label}: <2 clusters - point estimate "
                  f"{_rmse(gr[f'e_{a}']) - _rmse(gr[f'e_{b}']):+.2%} only.")
            continue
        m, lo, hi, kk = out
        flag = '  (few clusters - CI not reliable)' if kk < MIN_CLUSTERS_CI else ''
        print(f"    {label}: {m:+.2%}  [{lo:+.2%}, {hi:+.2%}]  "
              f"({kk} clusters){flag}")

# ---------- one figure per salt: kappa vs T ----------
CURVE_STYLE = [
    ('mc',    'McCleskey et al. (2012)',  'tab:red',   '-',  1.6),
    ('h_raw', 'Holdout raw $\\lambda$',   'tab:blue',  '--', 1.5),
    ('h_wal', 'Holdout Walden',           'tab:green', '-',  1.8),
    ('f_raw', 'Full-fit raw $\\lambda$',  'tab:cyan',  '--', 1.5),
    ('f_wal', 'Full-fit Walden',          'tab:olive', '-',  1.8),
]
_grid_cols = list(ION_FREE_COL.values()) + PAIR_COLS + ['I_eff']
for salt in sorted(held):
    in_sample = salt in FIT_SALTS
    m_vals = np.round(held[salt]['Molality'].to_numpy(float), 6)
    m_star = float(pd.Series(m_vals).value_counts().index[0])
    d_amb = salt_dfs[salt][salt_dfs[salt]['Ambient']]
    ds = d_amb[np.round(d_amb['Molality'].to_numpy(float), 6) == m_star] \
        .sort_values('temp_K')
    if len(ds) < 3:
        print(f"{salt}: fewer than 3 ambient rows at m={m_star:g} - no figure.")
        continue
    T_lo = max(ds['temp_K'].min(), VISC.T_MIN)
    T_hi = min(ds['temp_K'].max(), 323.15)   # plot up to 50 C
    Tg = np.linspace(T_lo, T_hi, 250)
    mg = np.full_like(Tg, m_star)
    grid = pd.DataFrame({c: spec_lookup.get(salt, Tg, mg, c) for c in _grid_cols}) \
        .rename(columns={'I_eff': 'I'})
    okg = np.isfinite(grid['I'].to_numpy(float))
    grid, Tg = grid[okg].reset_index(drop=True), Tg[okg]
    if len(Tg) < 2:
        print(f"{salt}: no speciation along the m={m_star:g} isotherm grid - no figure.")
        continue
    curves = {'mc': mc12_kappa(grid, Tg)}
    curves.update({k: chain_kappa(mdls[k], grid, Tg) for k in VKEYS})
    T_C = Tg - 273.15
    floor_C = chain_floor_C(salt)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for key, name, color, ls, lw in CURVE_STYLE:
        if key.startswith('f_'):
            name += ' (in-sample)' if in_sample else ' (out-of-sample salt)'
        ax.plot(T_C, curves[key], ls, color=color, lw=lw, label=name)
    ds_p = ds[ds['temp_K'] <= T_hi]
    ax.plot(ds_p['temp_K'] - 273.15, ds_p['Conductivity'], 'ko', ms=5,
            zorder=5, label='Measured')
    ax.axvline(floor_C, color='k', ls=':', lw=1.3)
    ax.axvline(0.0, color='tab:red', ls='-.', lw=1.0, alpha=0.7)
    if floor_C > T_C.min():
        ax.axvspan(T_C.min(), floor_C, color='0.93', zorder=0)
    ylo, yhi = ax.get_ylim()
    ax.text(floor_C - 0.5, yhi - 0.03 * (yhi - ylo),
            'holdout-chain extrapolation', ha='right', va='top', fontsize=8)
    ax.text(0.0, ylo + 0.03 * (yhi - ylo), ' MC12 range edge (0 C)',
            ha='left', va='bottom', fontsize=8, color='tab:red')
    ax.set_xlabel(r'T ($^\circ$C)')
    ax.set_ylabel(r'$\kappa$ (mS cm$^{-1}$)')
    role_txt = 'fit salt' if in_sample else 'held-out salt'
    ax.set_title(f'{salt} ({role_txt}), m = {m_star:g} mol kg$^{{-1}}$')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    save_fig(fig, f'cold_holdout_five_models_{salt}')
    plt.show()
    plt.close(fig)
