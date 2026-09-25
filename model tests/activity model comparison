# ==========================================================
# ACTIVITY MODEL (AND SPECIATION ARCHITECTURE) COMPARISON, FOR EACH SALT
# PER-VARIANT RETRAIN + OVERLAP-ENFORCED ISOPLETH FIGURES
# ==========================================================

import traceback
import warnings
import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm

_ORIG_SHOW = plt.show

GITHUB_RAW = 'https://raw.githubusercontent.com/alicjaurbanska/conductivity-jpl/main/'
MODEL_SOURCE_PATH = 'conductivity spline model'   # exact file name/path in the repo

VARIANTS = {
    'supcrt_hkf':    'SUPCRT / HKF',
    'supcrt_davies': 'SUPCRT / Davies',
    'supcrt_edh':    'SUPCRT / ext. Debye-Huckel',
    'wateq4f_bdot':  'WATEQ4F / B-dot',
    'pitzer':        'pitzer.dat / Pitzer',
}
VSTYLE = {
    'supcrt_hkf':    (0, (1, 1)),
    'supcrt_davies': (0, (5, 2)),
    'supcrt_edh':    (0, (3, 1, 1, 1)),
    'wateq4f_bdot':  'solid',
    'pitzer':        (0, (5, 1, 1, 1, 1, 1)),
}
MOL_ROUND = 3
NG = 60                       # points in the common T grid
MIN_PTS_CURVE = 2             # groups with fewer in-window points: scored
                              # if possible, but no curve is drawn

# Per-salt x-axis override: salt -> reference isopleth molality. Display only.
XLIM_REF_ISO = {'MgSO4': 0.101}
XLIM_PAD_K = 2.0

# ---- fetch model source and patch the variant lines -------------------
print(f'Fetching model source: {MODEL_SOURCE_PATH}')
resp = requests.get(GITHUB_RAW + quote(MODEL_SOURCE_PATH))
if resp.status_code != 200:
    raise RuntimeError(f'Could not fetch {MODEL_SOURCE_PATH} from GitHub '
                       f'(HTTP {resp.status_code}). Check the file name and path.')
source = resp.text

import re
_DB_RE = re.compile(r"^SPECIATION_DB = '[^']*'.*$", re.M)   # any production DB
_FIG_LINE = "FIG_DIR = 'Figures'"
_MISS_LINE = "_missing = df['I_eff'].isna()\n"
if not _DB_RE.search(source):
    raise RuntimeError('Model source on GitHub has no SPECIATION_DB line -- '
                       'push the current model cell.')
for _line in (_FIG_LINE, _MISS_LINE):
    if _line not in source:
        raise RuntimeError(f'Model source on GitHub has no line {_line!r} -- '
                           f'push the current model cell.')

# Non-ambient rows are never used in this ambient comparison, so a variant's
# speciation table may lack them. Drop such rows before the model's merge
# check; the check itself still runs on every ambient row.
_DROP_NONAMBIENT_GAPS = (
    "_nonamb_gap = (df['I_eff'].isna() & (df['Source'] != 'McCleskey2011') &\n"
    "               ((df['Pressure_MPa'] - REFERENCE_PRESSURE_MPA).abs() > AMBIENT_P_TOL_MPA))\n"
    "if _nonamb_gap.any():\n"
    "    print(f'[comparison] dropping {int(_nonamb_gap.sum())} non-ambient row(s) '\n"
    "          f'with no speciation in {SPECIATION_FILE} (not used at ambient P).')\n"
    "    df = df[~_nonamb_gap].copy()\n"
)

def variant_source(key):
    s = _DB_RE.sub(f"SPECIATION_DB = '{key}'", source, count=1)
    s = s.replace(_FIG_LINE, f"FIG_DIR = 'Figures_{key}'", 1)
    return s.replace(_MISS_LINE, _DROP_NONAMBIENT_GAPS + _MISS_LINE, 1)

# ---- fetch the McCleskey (2012) baseline -----------------------------
MC12_SOURCE_PATH = 'MC12 conductivity model'
print(f'Fetching MC12 source: {MC12_SOURCE_PATH}')
resp_mc = requests.get(GITHUB_RAW + quote(MC12_SOURCE_PATH))
if resp_mc.status_code != 200:
    raise RuntimeError(f'Could not fetch {MC12_SOURCE_PATH} from GitHub '
                       f'(HTTP {resp_mc.status_code}). Check the file name and path.')
plt.show = lambda *a, **k: None          # silence any figures in the MC12 file
_mc12_ns = {'__name__': 'mc12_conductivity_model'}
exec(compile(resp_mc.text, MC12_SOURCE_PATH, 'exec'), _mc12_ns)
plt.close('all')
if 'mccleskey2012_conductivity' not in _mc12_ns:
    raise RuntimeError(f'{MC12_SOURCE_PATH} defines no function '
                       f'mccleskey2012_conductivity.')
mccleskey2012_conductivity = _mc12_ns['mccleskey2012_conductivity']

# ---- retrain loop ----------------------------------------------------
trained, failed = {}, {}
plt.show = lambda *a, **k: None          # silence model figures during retrains
for key, label in VARIANTS.items():
    print('\n' + '#' * 70)
    print(f'# RETRAIN: {key}  ({label})')
    print('#' * 70)
    g = {'__name__': f'conductivity_model_{key}'}
    try:
        exec(compile(variant_source(key), f'conductivity_model[{key}]', 'exec'), g)
        trained[key] = g
        print(f'\n[{key}] retrain OK.')
    except Exception as e:
        failed[key] = traceback.format_exc()
        print(f'\n[{key}] RETRAIN FAILED: {e}')
    plt.close('all')
plt.show = _ORIG_SHOW

if failed:
    print('\nFailed variants (a gate tripping IS a result):')
    for k, tb in failed.items():
        print(f'\n--- {k} ---\n{tb.splitlines()[-1]}')
if not trained:
    raise RuntimeError('No variant trained successfully.')
vkeys = list(trained)

# ---- prediction wrappers ---------------------------------------------
def _pull(g, salt, T_K, m, col):
    return np.nan_to_num(np.atleast_1d(g['spec_lookup'].get(salt, T_K, m, col)))

def predict_mc12(g, salt, T_K, m):
    T_K = np.atleast_1d(np.asarray(T_K, float))
    m = np.atleast_1d(np.asarray(m, float))
    species = {}
    for ion, col in g['ION_FREE_COL'].items():
        v = _pull(g, salt, T_K, m, col)
        if np.any(v > 0):
            species[col] = v
    for col in g['PAIR_COLS']:
        v = _pull(g, salt, T_K, m, col)
        if np.any(v > 0):
            species[col] = v
    I = np.atleast_1d(g['spec_lookup'].get(salt, T_K, m, 'I_eff'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.atleast_1d(mccleskey2012_conductivity(
            species, T_K - 273.15, I_eff_molkg=I,
            validate_range=False).sigma_mScm)

def predict_thiswork(g, salt, T_K, m):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.atleast_1d(g['model'].predict_from_speciation(
            salt, np.atleast_1d(np.asarray(T_K, float)),
            np.atleast_1d(np.asarray(m, float)), extrap_mode='linear'))

SCORERS = {'this work': predict_thiswork, 'MC12': predict_mc12}

def safe_pred(fn, g, salt, T, m):
    try:
        p = fn(g, salt, T, m)
        return np.where(np.isfinite(p), p, np.nan)
    except Exception:
        return np.full(np.atleast_1d(T).shape, np.nan)

# ---- isopleth registry: ALL molality groups per salt -----------------
ref = trained[vkeys[0]]
PLOT_SALTS = [s for s in ['NaCl', 'MgSO4', 'NH4Cl', 'KCl', 'Na2SO4',
                          'K2SO4', 'CaCl2', 'CaSO4', 'MgCl2']
              if s in ref['salt_dfs'] and not ref['salt_dfs'][s].empty]

iso_registry = {}     # salt -> list of dicts (m_grp, m_eval, data, cov)
for salt in PLOT_SALTS:
    d = ref['salt_dfs'][salt]
    d = d[d['Ambient']].copy()
    if d.empty:
        continue
    d['m_grp'] = d['Molality'].round(MOL_ROUND)
    entries = []
    for m0 in sorted(d['m_grp'].unique()):
        sel = d[d['m_grp'] == m0]
        m_eval = float(sel['Molality'].mean())
        # ---- per-variant coverage at each point's OWN (T, m) ----
        # A point counts for variant k if BOTH models (this work, MC12) give
        # a finite prediction with k's speciation, so the two models are
        # always compared on the same points within a variant.
        T_pts = sel['temp_K'].values
        m_pts = sel['Molality'].values
        cov = {}
        for k in vkeys:
            ok = np.ones(len(sel), dtype=bool)
            for name, fn in SCORERS.items():
                ok &= np.isfinite(safe_pred(fn, trained[k], salt, T_pts, m_pts))
            cov[k] = ok
        anycov = np.any(list(cov.values()), axis=0)
        if not anycov.any():
            print(f'  [{salt} @ {m0:g}] no variant covers any point -- isopleth skipped')
            continue
        missing = [f'{k} {int(cov[k].sum())}/{len(sel)}'
                   for k in vkeys if cov[k].sum() < len(sel)]
        if missing:
            print(f'  [{salt} @ {m0:g}] partial coverage: ' + ', '.join(missing))
        entries.append(dict(m_grp=m0, m_eval=m_eval, data=sel,
                            cov=cov, anycov=anycov))
    if entries:
        iso_registry[salt] = entries

# ---- curve grid per variant: T over that variant's covered points ------
def curve_grid(e, k):
    """T grid over variant k's covered points; molality follows the measured
    molality of those points (averaged per T), so the curve stays inside the
    speciation table. Returns None if fewer than MIN_PTS_CURVE temperatures."""
    sc = (e['data'][e['cov'][k]].groupby('temp_K', as_index=False)['Molality']
          .mean().sort_values('temp_K'))
    if len(sc) < MIN_PTS_CURVE:
        return None
    T_grid = np.linspace(sc['temp_K'].min(), sc['temp_K'].max(), NG)
    m_grid = np.interp(T_grid, sc['temp_K'].values, sc['Molality'].values)
    return T_grid, m_grid

# ---- isopleth colors: viridis ordered by molality per salt -----------
def iso_color(salt, e):
    entries = iso_registry[salt]
    ms = [x['m_grp'] for x in entries]
    if len(ms) == 1:
        return cm.viridis(0.5)
    rank = sorted(ms).index(e['m_grp'])
    return cm.viridis(0.08 + 0.84 * rank / (len(ms) - 1))

# ---- score each variant on its own covered points -------------------
rows = []
for salt, entries in iso_registry.items():
    for e in entries:
        for k in vkeys:
            sel = e['data'][e['cov'][k]]
            if sel.empty:
                continue
            T = sel['temp_K'].values
            m = sel['Molality'].values
            meas = sel['Conductivity'].values
            for name, fn in SCORERS.items():
                pred = safe_pred(fn, trained[k], salt, T, m)
                ok = np.isfinite(pred) & (meas > 0)
                if ok.any():
                    dev = (pred[ok] - meas[ok]) / meas[ok]
                    rows.append(dict(
                        salt=salt, isopleth=e['m_grp'], variant=k, model=name,
                        n=int(ok.sum()),
                        rel_rmse=100 * float(np.sqrt(np.mean(dev ** 2))),
                        mean_bias=100 * float(np.mean(dev))))
st = pd.DataFrame(rows)
if st.empty:
    raise RuntimeError('No isopleth could be scored: every prediction was NaN. '
                       'A scorer is failing silently inside safe_pred -- call '
                       'the scorers directly on a few rows to see the error.')

print('\n' + '=' * 70)
print('rel. RMSE (%) per variant on that variant\'s OWN covered points')
print('(both models on the same points within a variant; point sets can')
print('differ between variants -- see the n table)')
print('=' * 70)
st['_w'] = st['n'] * (st['rel_rmse'] / 100) ** 2
for name in SCORERS:
    sub = st[st['model'] == name]
    g = sub.groupby(['salt', 'variant'])[['_w', 'n']].sum()
    agg = (100 * np.sqrt(g['_w'] / g['n'])).unstack()
    print(f'\n--- {name} ---')
    print(agg.round(2).to_string())
n_tab = (st[st['model'] == 'this work'].groupby(['salt', 'variant'])['n']
         .sum().unstack().fillna(0).astype(int))
n_all = pd.Series({s: sum(len(e['data']) for e in es)
                   for s, es in iso_registry.items()}, name='total')
print('\n--- n scored points per variant (total ambient points in last column) ---')
print(n_tab.join(n_all).to_string())
st = st.drop(columns='_w')
st.to_csv('comparison_v4_common.csv', index=False)

# ---- shared y-limits per salt (across both figures) ------------------
ylims = {}
for salt, entries in iso_registry.items():
    vals = []
    for e in entries:
        vals.append(e['data']['Conductivity'].values / 10.0)
        for k in vkeys:
            cg = curve_grid(e, k)
            if cg is None:
                continue
            for name, fn in SCORERS.items():
                p = safe_pred(fn, trained[k], salt, *cg) / 10.0
                vals.append(p[np.isfinite(p)])
    allv = np.concatenate([v for v in vals if len(v)])
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ylims[salt] = (max(0, lo - pad), hi + pad)

# ---- per-salt x-limit overrides (display only) -----------------------
xlims = {}
for salt, ref_m in XLIM_REF_ISO.items():
    entries = iso_registry.get(salt)
    if not entries:
        continue
    e_ref = min(entries, key=lambda e: abs(e['m_grp'] - ref_m))
    T_all = e_ref['data']['temp_K'].values
    xlims[salt] = (float(T_all.min()) - XLIM_PAD_K,
                   float(T_all.max()) + XLIM_PAD_K)
    print(f"[xlim] {salt}: trimmed to {e_ref['m_grp']:g} mol/kg span "
          f'{xlims[salt][0]:.1f}-{xlims[salt][1]:.1f} K (display only)')

# ---- figures: absolute panel + residual strip per salt ---------------
def make_figure(model_name, fname):
    fn = SCORERS[model_name]
    salts = list(iso_registry)
    ncols = 4
    nrows = int(np.ceil(len(salts) / ncols))
    fig = plt.figure(figsize=(4.5 * ncols, 4.6 * nrows))
    outer = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.55, wspace=0.32)
    for i, salt in enumerate(salts):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i // ncols, i % ncols],
            height_ratios=[2.4, 1.0], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        n_scored = 0
        for e in iso_registry[salt]:
            col = iso_color(salt, e)
            sel, inwin = e['data'], e['anycov']
            ax.scatter(sel['temp_K'][inwin], sel['Conductivity'][inwin] / 10.0,
                       s=24, color=col, edgecolor='k', linewidth=0.4,
                       zorder=5, label=f"{e['m_grp']:g}")
            if (~inwin).any():
                ax.scatter(sel['temp_K'][~inwin],
                           sel['Conductivity'][~inwin] / 10.0,
                           s=20, facecolor='none', edgecolor='0.6',
                           linewidth=0.8, zorder=4)
            n_scored += int(inwin.sum())
            for k in vkeys:
                cg = curve_grid(e, k)
                if cg is None:
                    continue
                pred = safe_pred(fn, trained[k], salt, *cg)
                ax.plot(cg[0], pred / 10.0, color=col,
                        linestyle=VSTYLE[k], linewidth=1.2, alpha=0.85)
            for k in vkeys:
                ck = e['cov'][k]
                Tp = sel['temp_K'][ck].values
                mp = sel['Molality'][ck].values
                msd = sel['Conductivity'][ck].values
                pred = safe_pred(fn, trained[k], salt, Tp, mp)
                ok = np.isfinite(pred) & (msd > 0)
                if ok.any():
                    axr.plot(Tp[ok], 100 * (pred[ok] - msd[ok]) / msd[ok],
                             color=col, linestyle=VSTYLE[k],
                             marker='.', ms=3, linewidth=0.8, alpha=0.8)
        ax.set_ylim(*ylims[salt])
        if salt in xlims:
            ax.set_xlim(*xlims[salt])
        ax.set_title(f'{salt}  (n={n_scored} scored)', fontsize=10)
        ax.set_ylabel(r'$\sigma$ (S/m)')
        ax.legend(fontsize=5, title='mol/kg', title_fontsize=5, ncol=2,
                  handletextpad=0.3, columnspacing=0.7, borderpad=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)
        axr.axhline(0, color='k', lw=0.7)
        axr.set_ylabel('dev (%)', fontsize=8)
        axr.set_xlabel('T (K)')
        axr.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color='0.3', linestyle=VSTYLE[k],
                          linewidth=1.6, label=VARIANTS[k]) for k in vkeys]
    handles.insert(0, plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='0.4', markeredgecolor='k',
                                 markersize=6, label='experiment (scored)'))
    handles.insert(1, plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='none',
                                 markeredgecolor='0.6', markersize=6,
                                 label='experiment (no variant covers it)'))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 4), fontsize=8, frameon=False)
    fig.suptitle('Binary-solution conductivity: speciation variants on a '
                 f'own coverage, scored with {model_name}', fontsize=13)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.09, top=0.93)
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png', dpi=300)
    plt.show()
    plt.close(fig)

make_figure('this work', 'isopleths_thiswork')
make_figure('MC12', 'isopleths_MC12')

print('\nNOTES:')
print(' 1. ALL molality groups are shown. Colors: viridis by molality')
print('    (dark = dilute, bright = concentrated).')
print(' 2. Each variant is scored on the points where BOTH models give a')
print('    finite prediction with that variant\'s speciation. A variant that')
print('    does not cover a point (e.g. Pitzer at high I) simply skips it;')
print('    the others still score it. Point counts are in the n table.')
print(' 3. Filled points are scored by at least one variant; hollow grey')
print('    points are covered by no variant. Each variant\'s curve spans its')
print('    own covered temperatures, with molality following the data.')
print(' 4. Because point sets differ between variants, compare RMSE across')
print('    variants together with n; this work vs MC12 within one variant is')
print('    always a same-points comparison.')
print(' 5. Pitzer uses a different activity framework (specific ion')
print('    interaction, few explicit ion pairs), so its free-ion / pair split')
print('    is not directly comparable with the ion-association variants.')
print(' 6. MgSO4 x-axis trimmed to the 0.101 mol/kg span (XLIM_REF_ISO).')
