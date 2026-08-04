# ============================================================
# DIAGNOSTIC 1 (COMMON SUPPORT + TEMPERATURE REGIMES)
# spline model vs McCleskey (2012), identical samples for both models.
# Run AFTER the model cell (v7) and AFTER the McCleskey cell.
#
# Metrics are split into three temperature regimes per salt:
#   (1) total     - all common-support points
#   (2) 5-95 C    - MC12's validated range
#   (3) < 5 C     - low temperature (extrapolation territory for MC12)
# Points above 95 C (McCleskey hot rows) are counted in 'total' only and
# reported separately so nothing vanishes silently.
#
# One panel per salt (unchanged from the common-support version):
#   dots = data (grey open = spline cannot evaluate, excluded everywhere),
#   dashed = spline, dotted = MC12 (masked to spline support),
#   colors = molality. ONE figure -> PDF + PNG via save_fig.
# ============================================================
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

_D1_M_ROUND = 6
_T_REGIME_LO_C = 5.0    # lower edge of MC12's validated range
_T_REGIME_HI_C = 95.0   # upper edge of MC12's validated range

# ---------- prediction helpers ----------
def _d1_mc12_kappa(salt, T_K, m):
    """MC12 kappa (mS/cm) at (T, m), fed by the SAME speciation table
    the spline model uses, so the comparison isolates the lambda models."""
    T_K = np.atleast_1d(np.asarray(T_K, float))
    m = np.broadcast_to(np.asarray(m, float), T_K.shape)
    species = {}
    for col in list(ION_FREE_COL.values()) + PAIR_COLS:
        v = spec_lookup.get(salt, T_K, m, col)
        if np.any(np.isfinite(v)):
            species[col] = np.nan_to_num(v)
    I = spec_lookup.get(salt, T_K, m, 'I_eff')
    if not species:
        return np.full(T_K.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = mccleskey2012_conductivity(species, T_K - 273.15,
                                         I_eff_molkg=np.nan_to_num(I),
                                         validate_range=False)
    return np.where(np.isfinite(I), res.sigma_mScm, np.nan)

def _d1_spline_kappa(salt, T_K, m):
    """Spline-model kappa (mS/cm) via the salt-keyed adapter (ambient)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.atleast_1d(model.predict_from_speciation(
            salt, np.atleast_1d(np.asarray(T_K, float)),
            np.atleast_1d(np.asarray(m, float)), extrap_mode='linear'))

def _rmse_bias(rel):
    if rel.size == 0:
        return np.nan, np.nan
    return float(np.sqrt(np.mean(rel ** 2))), float(np.mean(rel))

# ---------- assemble ALL empirical data, per salt ----------
_d1_frames = {}
for _salt, _d in salt_dfs.items():
    _d1_frames.setdefault(_salt, []).append(_d)
for (_salt, _src), _d in cross_lab_validation_dfs.items():
    _d1_frames.setdefault(_salt, []).append(_d)
if len(mixture_validation_df):
    for _salt, _d_raw in mixture_validation_df.groupby('Salt'):
        try:
            _d = _clean_speciated_rows(_d_raw, f'{_salt} (diag1)',
                                       require_ambient=False)
            _d = _iapws_filter(_d, f'{_salt} (diag1)')
            _d1_frames.setdefault(_salt, []).append(_d)
        except Exception as _e:
            print(f'diag1: skipping mixture {_salt}: {_e}')

diag1_data = {}
for _salt, _parts in _d1_frames.items():
    _d = pd.concat(_parts, ignore_index=True).drop_duplicates(
        subset=['Source', 'temp_K', 'Molality', 'Pressure_MPa'])
    _n_amb = int(_d['Ambient'].sum())
    if _n_amb < len(_d):
        print(f'diag1: {_salt}: {len(_d) - _n_amb} non-ambient row(s) excluded '
              f'(both models are ambient/0.1 MPa frameworks).')
    _d = _d[_d['Ambient']].copy()
    if _d.empty:
        continue
    if _salt not in spec_lookup.interp2d:
        print(f'diag1: {_salt}: no speciation table entry - skipped.')
        continue
    diag1_data[_salt] = _d.sort_values(['Molality', 'temp_K']).reset_index(drop=True)

# ---------- metrics + figure ----------
print('\n' + '=' * 70)
print('DIAGNOSTIC 1 (COMMON SUPPORT, T-REGIME SPLIT): spline vs MC12')
print('Both models evaluated on IDENTICAL samples (spline-evaluable points).')
print('=' * 70)

_salts = list(diag1_data)
_ncols = 3
_nrows = int(np.ceil(len(_salts) / _ncols))
fig, axes = plt.subplots(_nrows, _ncols, figsize=(5.2 * _ncols, 4.0 * _nrows),
                         squeeze=False)
_summary = []   # rows: (salt, regime, n, rmse_s, bias_s, rmse_m, bias_m)
for _i, _salt in enumerate(_salts):
    ax = axes[_i // _ncols][_i % _ncols]
    d = diag1_data[_salt]
    T_K = d['temp_K'].values
    T_C = T_K - 273.15
    M = d['Molality'].values
    kappa = d['Conductivity'].values

    try:
        p_spl = _d1_spline_kappa(_salt, T_K, M)
    except Exception as _e:
        p_spl = np.full_like(kappa, np.nan)
        print(f'diag1: {_salt}: spline prediction failed ({_e}).')
    p_mc = _d1_mc12_kappa(_salt, T_K, M)

    common = np.isfinite(p_spl) & np.isfinite(p_mc) & \
        np.isfinite(kappa) & (kappa != 0)
    n_all, n_common = len(d), int(common.sum())
    if n_all - n_common:
        _out = ~common
        print(f'  {_salt}: {n_all - n_common}/{n_all} point(s) outside common '
              f'support, T {T_C[_out].min():.1f} to {T_C[_out].max():.1f} C, '
              f'm {M[_out].min():.3g} to {M[_out].max():.3g} mol/kg - '
              f'EXCLUDED from BOTH models everywhere below.')

    rel_s = np.where(common, (p_spl - kappa) / kappa, np.nan)
    rel_m = np.where(common, (p_mc - kappa) / kappa, np.nan)

    # ---- temperature regimes (within common support) ----
    reg_total = common
    reg_valid = common & (T_C >= _T_REGIME_LO_C) & (T_C <= _T_REGIME_HI_C)
    reg_cold = common & (T_C < _T_REGIME_LO_C)
    reg_hot = common & (T_C > _T_REGIME_HI_C)   # counted in total only
    if reg_hot.any():
        print(f'  {_salt}: {int(reg_hot.sum())} common-support point(s) ABOVE '
              f'{_T_REGIME_HI_C:.0f} C (up to {T_C[reg_hot].max():.0f} C) - '
              f'included in TOTAL only, not in either sub-regime.')
    for regime_name, msk in [('total', reg_total),
                             (f'{_T_REGIME_LO_C:.0f}-{_T_REGIME_HI_C:.0f} C', reg_valid),
                             (f'< {_T_REGIME_LO_C:.0f} C', reg_cold)]:
        rmse_s, bias_s = _rmse_bias(rel_s[msk])
        rmse_m, bias_m = _rmse_bias(rel_m[msk])
        _summary.append((_salt, regime_name, int(msk.sum()),
                         rmse_s, bias_s, rmse_m, bias_m))
    rmse_s_tot = _summary[-3][3]
    rmse_m_tot = _summary[-3][5]

    # ---- panel (unchanged plotting) ----
    m_levels = np.unique(np.round(M, _D1_M_ROUND))
    if len(m_levels) > 1:
        norm = mcolors.Normalize(vmin=m_levels.min(), vmax=m_levels.max())
    else:
        norm = mcolors.Normalize(vmin=m_levels[0] * 0.9, vmax=m_levels[0] * 1.1 + 1e-9)
    cmap = plt.cm.viridis
    for m0 in m_levels:
        sel = np.round(M, _D1_M_ROUND) == m0
        c = cmap(norm(m0))
        in_sup = sel & common
        out_sup = sel & ~common
        ax.scatter(T_C[in_sup], kappa[in_sup], color=c, s=16,
                   zorder=3, edgecolors='none')
        ax.scatter(T_C[out_sup], kappa[out_sup], facecolors='none',
                   edgecolors='0.6', s=16, lw=0.8, zorder=3)
        T_lo, T_hi = T_K[sel].min(), T_K[sel].max()
        if T_hi - T_lo < 1.0:
            T_lo, T_hi = T_lo - 3.0, T_hi + 3.0
        Tg = np.linspace(T_lo, T_hi, 120)
        mg = np.full_like(Tg, m0)
        try:
            y_spl = _d1_spline_kappa(_salt, Tg, mg)
        except Exception:
            y_spl = np.full_like(Tg, np.nan)
        y_mc = _d1_mc12_kappa(_salt, Tg, mg)
        y_mc = np.where(np.isfinite(y_spl), y_mc, np.nan)
        if np.any(np.isfinite(y_spl)):
            ax.plot(Tg - 273.15, y_spl, '--', color=c, lw=1.4, zorder=2)
        if np.any(np.isfinite(y_mc)):
            ax.plot(Tg - 273.15, y_mc, ':', color=c, lw=1.4, zorder=2)
    ax.axvline(_T_REGIME_LO_C, color='steelblue', lw=0.8)
    _xlo = ax.get_xlim()[0]
    if T_C[common].size and T_C[common].min() < _T_REGIME_LO_C:
        ax.axvspan(_xlo, _T_REGIME_LO_C, color='steelblue', alpha=0.08, zorder=0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label('molality (mol kg$^{-1}$)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel('T ($^\\circ$C)')
    ax.set_ylabel('$\\kappa$ (mS cm$^{-1}$)')
    ax.set_title(f'{_salt} (n={n_common}/{n_all} common)\n'
                 f'spline rel. RMSE {rmse_s_tot:.1%} | MC12 rel. RMSE {rmse_m_tot:.1%}',
                 fontsize=9)

for _j in range(len(_salts), _nrows * _ncols):
    axes[_j // _ncols][_j % _ncols].set_visible(False)

_style_handles = [
    Line2D([], [], color='k', marker='o', ls='none', ms=5,
           label='data (common support)'),
    Line2D([], [], color='0.6', marker='o', mfc='none', ls='none', ms=5,
           label='data (spline cannot evaluate - excluded)'),
    Line2D([], [], color='k', ls='--', label='spline model'),
    Line2D([], [], color='k', ls=':', label='McCleskey 2012 (same support)'),
]
fig.legend(handles=_style_handles, loc='upper right', ncol=2, frameon=False)
fig.suptitle('Diagnostic 1 (common support): spline model vs McCleskey (2012) '
             f'({FIT_MODE} fit, speciation={SPECIATION_DB}); shaded = below '
             f'{_T_REGIME_LO_C:.0f} C', y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, 'diag1_spline_vs_mc12_common_support_regimes')
plt.show()
plt.close(fig)

# ---------- regime-split summary table ----------
print('\nRelative RMSE and mean signed bias (+ = overprediction), both models '
      'on the SAME common-support points, split by temperature regime:')
print(f"{'Salt':22s} {'regime':>9s} {'n':>5s}  "
      f"{'spline RMSE':>11s} {'spline bias':>11s}  "
      f"{'MC12 RMSE':>10s} {'MC12 bias':>10s}")
_prev_salt = None
for _salt, _reg, _n, _rs, _bs, _rm, _bm in _summary:
    if _prev_salt is not None and _salt != _prev_salt:
        print('-' * 92)
    _prev_salt = _salt
    if _n == 0:
        print(f'{_salt:22s} {_reg:>9s} {_n:5d}  {"-":>11s} {"-":>11s}  '
              f'{"-":>10s} {"-":>10s}')
    else:
        print(f'{_salt:22s} {_reg:>9s} {_n:5d}  {_rs:11.2%} {_bs:+11.2%}  '
              f'{_rm:10.2%} {_bm:+10.2%}')
print('\nReading guide: the 5-95 C rows are MC12 on home turf - its fairest '
      'showing. The < 5 C rows isolate extrapolation behavior for both '
      'frameworks (MC12: raw quadratics; spline: in-hull cold data or tangent '
      'extrapolation). A model whose total looks fine but whose regimes have '
      'opposite-sign biases is cancelling errors, not fitting well.')
