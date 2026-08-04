# ============================================================
# PAIRING CHECK: does WATEQ4F put more MgSO4 mass into the neutral
# pair than SUPCRT? Compares m_free_Mg / m_total at matched (T, m, P)
# keys for Salt == 'MgSO4'. Standalone: fetches both CSVs from GitHub.
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote

_RAW = 'https://raw.githubusercontent.com/alicjaurbanska/conductivity-jpl/main/'
_FILES = {'supcrt': 'speciation databases/speciation_supcrt.csv',
          'wateq4f': 'speciation databases/speciation_wateq4f.csv'}
_R = 6  # key rounding, same as MERGE_DECIMALS

tabs = {}
for db, path in _FILES.items():
    t = pd.read_csv(_RAW + quote(path))
    t = t[t['Salt'] == 'MgSO4'].copy()
    t['_t'] = t['Temperature_C'].round(_R)
    t['_m'] = t['Molality_molkg'].round(_R)
    t['_p'] = t['Pressure_MPa'].round(_R)
    t['free_frac'] = t['m_free_Mg'] / t['Molality_molkg']
    if 'm_pair_MgSO4' in t.columns:
        t['pair_frac'] = t['m_pair_MgSO4'] / t['Molality_molkg']
    print(f'{db:8s}: {len(t)} MgSO4 rows, T {t["Temperature_C"].min():.0f} to '
          f'{t["Temperature_C"].max():.0f} C, m {t["Molality_molkg"].min():.3g} '
          f'to {t["Molality_molkg"].max():.3g} mol/kg')
    tabs[db] = t

cols = ['_t', '_m', '_p', 'Temperature_C', 'Molality_molkg', 'free_frac',
        'I_eff'] + (['pair_frac'] if 'pair_frac' in tabs['supcrt'].columns else [])
mrg = tabs['supcrt'][cols].merge(
    tabs['wateq4f'][['_t', '_m', '_p', 'free_frac', 'I_eff'] +
                    (['pair_frac'] if 'pair_frac' in tabs['wateq4f'].columns else [])],
    on=['_t', '_m', '_p'], suffixes=('_sup', '_w4f'))
print(f'\nMatched (T, m, P) keys: {len(mrg)}')
if len(mrg) == 0:
    raise SystemExit('No matched keys - check rounding or table contents.')

d_free = mrg['free_frac_w4f'] - mrg['free_frac_sup']
print('\nFree-Mg fraction m_free_Mg / m_total (LOWER = MORE pairing):')
print(f'  SUPCRT : mean {mrg["free_frac_sup"].mean():.3f}, '
      f'range {mrg["free_frac_sup"].min():.3f} to {mrg["free_frac_sup"].max():.3f}')
print(f'  WATEQ4F: mean {mrg["free_frac_w4f"].mean():.3f}, '
      f'range {mrg["free_frac_w4f"].min():.3f} to {mrg["free_frac_w4f"].max():.3f}')
print(f'  WATEQ4F - SUPCRT: mean {d_free.mean():+.3f}, '
      f'range {d_free.min():+.3f} to {d_free.max():+.3f}; '
      f'WATEQ4F pairs MORE on {int((d_free < 0).sum())}/{len(mrg)} keys.')
print(f'\n  I_eff ratio WATEQ4F/SUPCRT: mean '
      f'{(mrg["I_eff_w4f"] / mrg["I_eff_sup"]).mean():.3f}')

# quick structure: free fraction by molality band (25 C-ish rows if present)
band = mrg[np.isclose(mrg['Temperature_C'], 25.0, atol=2.5)]
src = band if len(band) >= 4 else mrg
lbl = '~25 C rows' if len(band) >= 4 else 'all rows'
print(f'\nFree-Mg fraction vs molality ({lbl}):')
for m0, g in src.groupby(src['Molality_molkg'].round(3)):
    print(f'  m={m0:7.3f}: SUPCRT {g["free_frac_sup"].mean():.3f}   '
          f'WATEQ4F {g["free_frac_w4f"].mean():.3f}')

fig, ax = plt.subplots(figsize=(6.5, 4.5))
sc = ax.scatter(mrg['Molality_molkg'], mrg['free_frac_sup'],
                c=mrg['Temperature_C'], cmap='coolwarm', marker='o', s=22,
                label='SUPCRT', edgecolors='none')
ax.scatter(mrg['Molality_molkg'], mrg['free_frac_w4f'],
           c=mrg['Temperature_C'], cmap='coolwarm', marker='^', s=26,
           label='WATEQ4F', edgecolors='k', linewidths=0.3)
fig.colorbar(sc, ax=ax, label='T ($^\\circ$C)')
ax.set_xlabel('molality (mol kg$^{-1}$)')
ax.set_ylabel('free-Mg fraction  m$_{free,Mg}$ / m$_{total}$')
ax.set_title('MgSO4 pairing: SUPCRT (circles) vs WATEQ4F (triangles)\n'
             'lower = more mass in the neutral pair')
ax.legend()
fig.tight_layout()
try:
    save_fig(fig, 'pairing_check_supcrt_vs_wateq4f')  # if run inside the notebook
except NameError:
    fig.savefig('pairing_check_supcrt_vs_wateq4f.png', dpi=300)
plt.show()

verdict = ('CONFIRMED: WATEQ4F assigns MORE mass to the MgSO4 pair '
           '(lower free fraction), consistent with both models '
           'underpredicting more under WATEQ4F.'
           if d_free.mean() < 0 else
           'NOT confirmed: WATEQ4F pairs LESS than SUPCRT - the larger '
           'underprediction must come from the refit lambda surfaces, '
           'not the speciation. Revisit the diagnosis.')
print(f'\nVERDICT: {verdict}')
