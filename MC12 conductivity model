# =============================================================================
# McCleskey et al. (2012) electrical conductivity model
# =============================================================================
# Reference (primary source):
#   McCleskey, R.B., Nordstrom, D.K., Ryan, J.N., Ball, J.W. (2012),
#   "A new method of calculating electrical conductivity with applications
#   to natural waters", Geochimica et Cosmochimica Acta 77, 369-382.
#
# Model equations implemented:
#   Eq. (8):  lambda_i(T, I) = lambda0_i(T) - A_i(T) * sqrt(I) / (1 + B_i * sqrt(I))
#             with lambda0_i(T) and A_i(T) quadratic polynomials in T (deg C),
#             coefficients from Table 1. Units: mS kg cm^-1 mol^-1.
#   Eq. (9):  I = 0.5 * sum_i m_i * z_i^2   (EFFECTIVE ionic strength, computed
#             from the full SPECIATED composition, i.e. free ions + charged
#             ion pairs; one single value for the whole solution, shared by
#             every ion in Eq. 8).
#   Eq. (7):  kappa = sum_i lambda_i * m_i   [mS/cm]  ->  * 0.1  ->  [S/m]
#   Eq. (6):  t_i = lambda_i * m_i / sum_j lambda_j * m_j   (transport numbers)
#   Eq. (2):  kappa25 = kappa / (1 + alpha * (T - 25))      (optional helper)
#
# Stated validity (paper abstract / Sec. 5): T = 0-95 C, effective ionic
# strength 0.0004-0.7 mol/kg (equations fitted on data up to 1 mol/kg),
# pH 1-10, P = 0.1 MPa only. 
#
# Electronic Annex EA-1 (parameter derivation, implemented at the bottom of
# this cell):
#   Eq. (EA2): lambda_target = (kappa_measured - sum_known lambda_i * m_i)
#                              / m_target
#   Fitting algorithm: for each candidate B, linear-regress lambda against
#   x = sqrt(I)/(1 + B*sqrt(I)) at each temperature; pick the single B that
#   maximizes the mean R^2 across temperatures; then fit the intercepts
#   (lambda0) and negative slopes (A) as quadratic polynomials in T.
#
# =============================================================================

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np

# -----------------------------------------------------------------------------
# Table 1 of McCleskey et al. (2012).
# lam = (a2, a1, a0) for lambda0(T) = a2*T^2 + a1*T + a0    [mS kg cm^-1 mol^-1]
# A   = (a2, a1, a0) for A(T)       = a2*T^2 + a1*T + a0
# B   = empirical constant (dimensionless in (mol/kg)^-1/2)
# z   = species charge
#
# NOTE vs PlanetProfile reference implementation: all coefficients below match
# the PlanetProfile transcription EXCEPT Fe3+ A0, which the paper's Table 1
# gives as -22.18 (A_Fe3+ = -0.09676 T^2 + 20.76 T - 22.18); PlanetProfile
# has +22.18. See summary notes.
# -----------------------------------------------------------------------------
MCCLESKEY_2012_PARAMS: Dict[str, Dict[str, Any]] = {
    "K+":     dict(z=+1, lam=(0.003046, 1.261, 40.70),  A=(0.00535, 0.9316, 22.59),  B=1.5),
    "Na+":    dict(z=+1, lam=(0.003763, 0.877, 26.23),  A=(0.00027, 1.141, 32.07),   B=1.7),
    "H+":     dict(z=+1, lam=(-0.01414, 5.355, 224.2),  A=(-0.00918, 1.842, 39.23),  B=0.3),
    "Li+":    dict(z=+1, lam=(0.002628, 0.7079, 19.20), A=(0.00412, 0.4632, 13.71),  B=0.2),
    "Cs+":    dict(z=+1, lam=(0.003453, 1.249, 43.94),  A=(0.00646, 0.7023, 21.79),  B=1.3),
    "NH4+":   dict(z=+1, lam=(0.003341, 1.285, 39.04),  A=(0.00132, 0.607, 11.19),   B=0.3),
    "Ca+2":   dict(z=+2, lam=(0.009645, 1.984, 62.28),  A=(0.03174, 2.334, 132.3),   B=2.8),
    "Mg+2":   dict(z=+2, lam=(0.01068, 1.695, 57.16),   A=(0.02453, 1.915, 80.50),   B=2.1),
    "Ba+2":   dict(z=+2, lam=(0.01059, 2.090, 68.10),   A=(0.03127, 2.248, 93.91),   B=1.9),
    "Sr+2":   dict(z=+2, lam=(0.006649, 2.069, 61.63),  A=(0.00702, 0.9009, 33.41),  B=0.1),
    "SO4-2":  dict(z=-2, lam=(0.01037, 2.838, 82.37),   A=(0.03324, 5.889, 193.5),   B=2.6),
    "Cl-":    dict(z=-1, lam=(0.003817, 1.337, 40.99),  A=(0.00613, 0.9469, 22.01),  B=1.5),
    "F-":     dict(z=-1, lam=(0.002764, 1.087, 26.66),  A=(0.00178, 0.6202, 19.34),  B=0.5),
    "Br-":    dict(z=-1, lam=(0.000709, 1.477, 40.91),  A=(0.00251, 0.5398, 12.01),  B=0.1),
    "CO3-2":  dict(z=-2, lam=(-0.000326, 2.998, 64.03), A=(-0.00181, 5.542, 120.2),  B=2.3),
    "HCO3-":  dict(z=-1, lam=(0.000614, 0.9048, 21.14), A=(-0.00503, 0.8957, 10.97), B=0.1),
    "NO3-":   dict(z=-1, lam=(0.001925, 1.214, 39.90),  A=(0.00118, 0.5045, 23.31),  B=0.1),
    "OH-":    dict(z=-1, lam=(0.003396, 2.925, 121.3),  A=(0.00933, 0.1086, 35.90),  B=0.01),
    "Al+3":   dict(z=+3, lam=(0.02376, 3.227, 90.24),   A=(0.06484, 5.149, 76.79),   B=3.0),
    "Cu+2":   dict(z=+2, lam=(0.00818, 1.939, 53.26),   A=(0.0292, 6.745, 151.5),    B=8.0),
    "Fe+2":   dict(z=+2, lam=(0.009939, 1.878, 54.80),  A=(0.03997, 3.217, 164.5),   B=4.0),
    "Fe+3":   dict(z=+3, lam=(0.02077, 4.390, 82.42),   A=(-0.09676, 20.76, -22.18), B=4.0),
    "Mn+2":   dict(z=+2, lam=(0.01275, 2.109, 46.19),   A=(0.1071, 9.023, 135.4),    B=7.6),
    "Zn+2":   dict(z=+2, lam=(0.01249, 1.912, 48.20),   A=(0.08284, 5.188, 75.73),   B=7.0),
    "KSO4-":  dict(z=-1, lam=(-0.002439, 4.253, 129.7), A=(-0.01576, 6.210, 146.8),  B=1.3),
    "NaSO4-": dict(z=-1, lam=(0.002309, 5.459, 219.2),  A=(0.01454, 5.193, 253.6),   B=0.5),
    "HSO4-":  dict(z=-1, lam=(0.000927, 0.8337, 29.56), A=(0.02887, 0.873, 36.25),   B=7.0),
    "NaCO3-": dict(z=-1, lam=(0.00336, 3.845, 89.51),   A=(0.00061, 6.387, 141.7),   B=2.0),
}

# Charges for species commonly returned by speciation codes that have NO
# molal-conductivity parameters in Table 1. They contribute to the effective
# ionic strength (if charged) but contribute zero conductivity (a warning is
# raised for charged ones). Neutral pairs (z = 0) are legal inputs and are
# simply inert here - this is exactly how MgSO4(aq)/CaSO4(aq) suppress
# conductivity in the McCleskey framework.
AUX_SPECIES_CHARGES: Dict[str, int] = {
    "MgSO4": 0, "CaSO4": 0, "NaCl": 0, "KCl": 0, "MgCO3": 0, "CaCO3": 0,
    "NaHCO3": 0, "NaOH": 0, "HCl": 0, "NH4SO4-": -1, "MgCl+": +1,
    "CaCl+": +1, "MgOH+": +1, "CaOH+": +1, "MgHCO3+": +1, "CaHCO3+": +1,
    "NaCO3": -1,  # placeholder-safe duplicate; canonical NaCO3- is in Table 1
}

# Name normalization: accept PlanetProfile style ("K_p1", "SO4_m2"),
# PHREEQC/SUPCRT style ("K+", "SO4-2", "SO4--"), bare stems ("K", "SO4",
# "KSO4"), and this project's CSV columns ("m_free_K", "m_pair_KSO4").
_STEM_TO_CANONICAL = {
    "K": "K+", "NA": "Na+", "H": "H+", "LI": "Li+", "CS": "Cs+", "NH4": "NH4+",
    "CA": "Ca+2", "MG": "Mg+2", "BA": "Ba+2", "SR": "Sr+2", "SO4": "SO4-2",
    "CL": "Cl-", "F": "F-", "BR": "Br-", "CO3": "CO3-2", "HCO3": "HCO3-",
    "NO3": "NO3-", "OH": "OH-", "AL": "Al+3", "CU": "Cu+2", "FE": "Fe+2",
    "FE2": "Fe+2", "FE3": "Fe+3", "MN": "Mn+2", "ZN": "Zn+2",
    "KSO4": "KSO4-", "NASO4": "NaSO4-", "HSO4": "HSO4-", "NACO3": "NaCO3-",
    "MGSO4": "MgSO4", "CASO4": "CaSO4", "NACL": "NaCl", "KCL": "KCl",
    "MGCO3": "MgCO3", "CACO3": "CaCO3", "NH4SO4": "NH4SO4-",
    "MGCL": "MgCl+", "CACL": "CaCl+", "MGOH": "MgOH+", "CAOH": "CaOH+",
    "MGHCO3": "MgHCO3+", "CAHCO3": "CaHCO3+",
}


def canonical_species_name(name: str) -> Optional[str]:
    """Map a species label from any supported convention to its canonical name.

    Returns None if the label is not recognized. Explicit charge suffixes in
    the input (e.g. 'Fe+3', 'Fe_p3', 'SO4--') take precedence over defaults
    (bare 'Fe' maps to Fe+2, the dominant redox state in Table 1 usage).
    """
    s = name.strip()
    s = re.sub(r"^m_(free|pair)_", "", s, flags=re.IGNORECASE)  # CSV columns
    s = re.sub(r"\(aq\)$", "", s, flags=re.IGNORECASE)
    # PlanetProfile style: K_p1, SO4_m2
    mpp = re.fullmatch(r"([A-Za-z0-9]+)_([pm])(\d)", s)
    if mpp:
        stem, sign, mag = mpp.group(1).upper(), mpp.group(2), int(mpp.group(3))
        chg = mag if sign == "p" else -mag
    else:
        # Parse charge suffix from the end: '+2'/'-2', '++'/'--', '+'/'-'
        m_num = re.fullmatch(r"([A-Za-z0-9]+)([+-])(\d)", s)
        m_rep = re.fullmatch(r"([A-Za-z0-9]+?)([+-]+)", s)
        if m_num:
            stem = m_num.group(1).upper()
            chg = int(m_num.group(3)) * (1 if m_num.group(2) == "+" else -1)
        elif m_rep:
            stem = m_rep.group(1).upper()
            suf = m_rep.group(2)
            chg = len(suf) * (1 if suf[0] == "+" else -1)
        elif re.fullmatch(r"[A-Za-z0-9]+", s):
            stem, chg = s.upper(), None
        else:
            return None
    if stem == "FE" and chg is not None:
        stem = "FE3" if chg == 3 else "FE2"
    cand = _STEM_TO_CANONICAL.get(stem)
    if cand is None:
        return None
    return cand


def _species_charge(canonical: str) -> Optional[int]:
    if canonical in MCCLESKEY_2012_PARAMS:
        return MCCLESKEY_2012_PARAMS[canonical]["z"]
    if canonical in AUX_SPECIES_CHARGES:
        return AUX_SPECIES_CHARGES[canonical]
    return None


def effective_ionic_strength(species_molality_molkg: Dict[str, Any],
                             extra_charges: Optional[Dict[str, int]] = None,
                             extra_params: Optional[Dict[str, Dict]] = None):
    """Effective (speciated) ionic strength, Eq. (9): I = 0.5 * sum m_i z_i^2.

    Input molalities must be RAW speciated molalities in mol/kg. Do NOT
    pre-multiply by z^2 or by 0.5 - the charge weighting happens here, once,
    for the whole solution.
    """
    extra_charges = extra_charges or {}
    extra_params = extra_params or {}
    I = 0.0
    for name, m in species_molality_molkg.items():
        canon = canonical_species_name(name)
        z = None
        if name in extra_params:
            z = extra_params[name]["z"]
        elif name in extra_charges:
            z = extra_charges[name]
        elif canon is not None:
            z = _species_charge(canon)
        if z is None:
            raise ValueError(
                f"Species '{name}' not recognized and no charge supplied via "
                f"extra_charges; cannot compute ionic strength safely.")
        I = I + 0.5 * np.asarray(m, dtype=float) * z ** 2
    return I


@dataclass
class McCleskeyResult:
    sigma_Sm: np.ndarray                 # electrical conductivity, S/m
    sigma_mScm: np.ndarray               # same, mS/cm (paper's working unit)
    I_eff_molkg: np.ndarray              # effective ionic strength used
    T_C: np.ndarray
    lambda_i: Dict[str, np.ndarray]      # molal conductivities, mS kg/cm/mol
    transport_numbers: Dict[str, np.ndarray]
    ignored_species: list                # charged species with no Table-1 params
    metadata: Dict[str, Any] = field(default_factory=dict)


def mccleskey2012_conductivity(species_molality_molkg: Dict[str, Any],
                               T_C,
                               *,
                               I_eff_molkg=None,
                               extra_charges: Optional[Dict[str, int]] = None,
                               extra_params: Optional[Dict[str, Dict]] = None,
                               speciation_source: str = "SUPCRT",
                               P_MPa=None,
                               validate_range: bool = True) -> McCleskeyResult:
    """Electrical conductivity of an aqueous solution, McCleskey et al. (2012).

    Parameters
    ----------
    species_molality_molkg : dict
        RAW speciated molalities (mol/kg) of free ions and ion pairs, as
        produced by an upstream speciation calculation (SUPCRT-based,
        PHREEQC/WATEQ4F, ...). Keys may use any supported naming convention
        ('K+', 'K', 'K_p1', 'm_free_K', 'SO4-2', 'm_pair_KSO4', ...). Values
        may be scalars or arrays broadcastable against T_C.
        Do NOT pass total (formal) salt concentrations: the method is defined
        on speciated compositions (paper Sec. 3.2).
    T_C : scalar or array
        Temperature in deg C. Valid range 0-95 C.
    I_eff_molkg : optional
        Effective ionic strength override, e.g. taken directly from the
        speciation code output (I_eff column). If None (default), it is
        computed internally from the full speciated composition via Eq. (9).
        This is the ONLY way to supply ionic strength; per-ion or
        pre-weighted quantities are not accepted.
    extra_charges : optional dict
        Charges for species not known to this module, so they can be included
        in the ionic strength (their conductivity contribution is zero).
    extra_params : optional dict
        User-fitted conductivity parameters for additional CHARGED species
        not in Table 1 (e.g. a newly fitted ion pair), keyed by the exact
        species name used in species_molality_molkg:
        {'MgCl+': {'z': 1, 'lam': (a2,a1,a0), 'A': (a2,a1,a0), 'B': b}}.
        Such species then contribute lambda*m to the conductivity (Eq. 7)
        and m*z^2/2 to the ionic strength. Typically produced by
        fit_mccleskey_parameters() below (EA-1 algorithm). Entries here take
        precedence over Table 1. Neutral species (z = 0) are rejected:
        consistent with McCleskey (2012), neutral pairs carry no current.
    speciation_source : str
        Metadata only (e.g. 'SUPCRT', 'WATEQ4F', 'PHREEQC'). Does not change
        the calculation; recorded in the result for provenance.
    P_MPa : optional
        Recorded as metadata only. The published model is calibrated at
        0.1 MPa and has NO pressure dependence; a warning is raised if a
        pressure other than ~0.1 MPa is supplied.
    validate_range : bool
        If True, warn when T or I_eff leave the published validity range.

    Returns
    -------
    McCleskeyResult
    """
    T_C = np.asarray(T_C, dtype=float)

    if P_MPa is not None and np.any(np.abs(np.asarray(P_MPa, float) - 0.1) > 1e-6):
        warnings.warn("McCleskey (2012) is calibrated at 0.1 MPa only; the "
                      "supplied P_MPa is recorded as metadata but no pressure "
                      "correction is applied.")

    # --- Effective ionic strength: one value per (T, composition) state -----
    if I_eff_molkg is None:
        I = effective_ionic_strength(species_molality_molkg, extra_charges,
                                     extra_params)
    else:
        I = np.asarray(I_eff_molkg, dtype=float)
        if np.any(I < 0):
            raise ValueError("I_eff_molkg must be non-negative.")
    I = np.broadcast_to(np.asarray(I, float), np.broadcast_shapes(np.shape(I), T_C.shape)).astype(float)

    if validate_range:
        if np.any(T_C < 0.0) or np.any(T_C > 95.0):
            warnings.warn("Temperature outside published range 0-95 C; "
                          "extrapolating Table-1 polynomials.")
        if np.any(I > 1.0):
            warnings.warn("Effective ionic strength > 1 mol/kg exceeds the "
                          "fitting range of McCleskey (2012).")
        elif np.any(I > 0.7):
            warnings.warn("Effective ionic strength > 0.7 mol/kg is beyond "
                          "the range validated against natural waters "
                          "(0.0004-0.7 mol/kg), though within the 1 mol/kg "
                          "fitting range.")

    sqI = np.sqrt(I)

    lam_out: Dict[str, np.ndarray] = {}
    contrib: Dict[str, np.ndarray] = {}
    ignored = []
    sigma_mScm = np.zeros(np.broadcast_shapes(T_C.shape, I.shape))

    extra_params = extra_params or {}
    for nm, pp in extra_params.items():
        if pp.get("z", 0) == 0:
            raise ValueError(
                f"extra_params['{nm}'] has z = 0. Neutral species carry no "
                "current in the McCleskey (2012) framework and cannot be "
                "assigned a conductivity contribution.")
    for name, m in species_molality_molkg.items():
        canon = canonical_species_name(name)
        m = np.asarray(m, dtype=float)
        if name in extra_params:
            p, canon = extra_params[name], name
        elif canon is not None and canon in MCCLESKEY_2012_PARAMS:
            p = MCCLESKEY_2012_PARAMS[canon]
        else:
            z = (extra_charges or {}).get(name,
                 AUX_SPECIES_CHARGES.get(canon) if canon else None)
            if z not in (0, None) and np.any(m > 0):
                ignored.append(name)
            continue
        lam0 = np.polyval(p["lam"], T_C)                       # lambda0(T)
        At = np.polyval(p["A"], T_C)                           # A(T)
        lam = lam0 - At * sqI / (1.0 + p["B"] * sqI)           # Eq. (8)
        if np.any(lam < 0):
            warnings.warn(f"Negative molal conductivity computed for {canon}; "
                          "outside reliable range of Eq. (8).")
        lam_out[canon] = lam
        contrib[canon] = lam * m                               # Eq. (7) term
        sigma_mScm = sigma_mScm + contrib[canon]

    if ignored:
        warnings.warn("Charged species with no Table-1 conductivity "
                      f"parameters were included in I but contribute zero "
                      f"conductivity: {sorted(set(ignored))}")

    total = np.where(sigma_mScm > 0, sigma_mScm, np.nan)
    t_num = {k: v / total for k, v in contrib.items()}         # Eq. (6)

    return McCleskeyResult(
        sigma_Sm=sigma_mScm * 0.1,        # 1 mS/cm = 0.1 S/m
        sigma_mScm=sigma_mScm,
        I_eff_molkg=I,
        T_C=T_C,
        lambda_i=lam_out,
        transport_numbers=t_num,
        ignored_species=sorted(set(ignored)),
        metadata=dict(model="McCleskey et al. (2012), GCA 77:369-382",
                      speciation_source=speciation_source,
                      P_MPa=P_MPa if P_MPa is not None else 0.1,
                      pressure_correction="none (model defined at 0.1 MPa)"),
    )


def kappa25(sigma, T_C, alpha=0.019):
    """Temperature-compensated conductivity, Eq. (2). Works in any unit of
    sigma; alpha = 0.019 1/C is the linear factor used in the paper's checks."""
    return np.asarray(sigma, float) / (1.0 + alpha * (np.asarray(T_C, float) - 25.0))


# =============================================================================
# EA-1: derivation of ionic molal conductivity parameters for new species
# (McCleskey et al. 2012, Electronic Annex EA-1)
# =============================================================================
def lambda_from_mixture(kappa_mScm,
                        target_species: str,
                        m_target_molkg,
                        known_species_molality_molkg: Dict[str, Any],
                        T_C,
                        *,
                        target_z: int = None,
                        I_eff_molkg=None,
                        extra_charges: Optional[Dict[str, int]] = None,
                        extra_params: Optional[Dict[str, Dict]] = None):
    """Eq. (EA2) generalized: back out the molal conductivity of one species
    from a MEASURED mixture conductivity.

        lambda_target = (kappa - sum_known lambda_i * m_i) / m_target

    Parameters
    ----------
    kappa_mScm : measured conductivity of the mixture, mS/cm (multiply S/m
        by 10 to convert).
    target_species : name of the CHARGED species being derived
        (e.g. 'MgCl+'). Neutral pairs are rejected: per McCleskey (2012),
        they carry no current, so no conductivity can be attributed to them.
    m_target_molkg : speciated molality of the target species.
    known_species_molality_molkg : speciated molalities of ALL other species.
    T_C : temperature(s), deg C.
    target_z : charge of the target species (required, must be nonzero);
        used for the internal effective-ionic-strength computation.
    I_eff_molkg : optional override of the effective ionic strength of the
        FULL solution (known + target). If None, computed internally.

    Returns
    -------
    (lambda_target [mS kg/cm/mol], I_eff used, x = sqrt(I)/(1+B*sqrt(I))
    requires a B and is therefore left to the fitting step)
    """
    m_t = np.asarray(m_target_molkg, dtype=float)
    if target_z is None or target_z == 0:
        raise ValueError(
            "target_z must be a nonzero charge. Deriving a conductivity "
            "contribution for a neutral species (e.g. MgSO4(aq)) is not part "
            "of the McCleskey (2012) method: neutral pairs carry no current.")
    if np.any(m_t <= 0):
        raise ValueError("m_target_molkg must be > 0 to apply Eq. (EA2).")
    if I_eff_molkg is None:
        I = effective_ionic_strength(known_species_molality_molkg,
                                     extra_charges, extra_params)
        I = I + 0.5 * m_t * target_z ** 2
    else:
        I = np.asarray(I_eff_molkg, dtype=float)
    res = mccleskey2012_conductivity(known_species_molality_molkg, T_C,
                                     I_eff_molkg=I,
                                     extra_charges=extra_charges,
                                     extra_params=extra_params,
                                     validate_range=False)
    lam_t = (np.asarray(kappa_mScm, dtype=float) - res.sigma_mScm) / m_t
    return lam_t, I


def fit_mccleskey_parameters(T_C, I_eff_molkg, lam, z: int,
                             B_grid=None, min_points_per_T: int = 3):
    """EA-1 fitting algorithm: turn (T, I, lambda) observations into Table-1
    style parameters {z, lam, A, B} usable via `extra_params`.

    Steps (verbatim from EA-1):
      1. For a candidate B, at each distinct temperature, linearly regress
         lambda against x = sqrt(I) / (1 + B*sqrt(I)).
      2. Choose the single, temperature-independent B that maximizes the mean
         R^2 over all per-temperature regressions.
      3. Fit intercepts lambda0(T) and A(T) = -slope(T) with second-order
         polynomials in T.

    Parameters
    ----------
    T_C, I_eff_molkg, lam : 1-D arrays of equal length. lam in
        mS kg cm^-1 mol^-1 (e.g. from lambda_from_mixture / Eq. EA2).
    z : charge of the species (must be nonzero; the EA-1 procedure applies
        to charged species only).
    B_grid : candidate B values; default 0.01 to 10 (log-spaced + paper-like
        values), covering the full range that appears in Table 1 (0.01-8).

    Returns
    -------
    dict(z=..., lam=(a2,a1,a0), A=(a2,a1,a0), B=..., diagnostics=...)
    """
    T_C = np.asarray(T_C, float).ravel()
    if int(z) == 0:
        raise ValueError("z must be nonzero: the EA-1 fitting procedure is "
                         "for charged species only (neutral pairs carry no "
                         "current in the McCleskey 2012 framework).")
    I = np.asarray(I_eff_molkg, float).ravel()
    lam = np.asarray(lam, float).ravel()
    if not (T_C.shape == I.shape == lam.shape):
        raise ValueError("T_C, I_eff_molkg, lam must have identical shapes.")
    if B_grid is None:
        B_grid = np.unique(np.concatenate([
            np.round(np.logspace(-2, 1, 200), 4), np.arange(0.1, 10.01, 0.1)]))

    Ts = np.unique(T_C)
    usable = [t for t in Ts if np.sum(T_C == t) >= min_points_per_T]
    if len(usable) < 3:
        raise ValueError("Need >= 3 temperatures with >= "
                         f"{min_points_per_T} points each to fit T-polynomials.")

    def per_T_regressions(B):
        r2s, intercepts, slopes = [], [], []
        for t in usable:
            sel = T_C == t
            x = np.sqrt(I[sel]) / (1.0 + B * np.sqrt(I[sel]))
            y = lam[sel]
            A1 = np.vstack([x, np.ones_like(x)]).T
            (slope, intercept), res_, *_ = np.linalg.lstsq(A1, y, rcond=None)
            yhat = slope * x + intercept
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)
            intercepts.append(intercept)
            slopes.append(slope)
        return np.array(r2s), np.array(intercepts), np.array(slopes)

    mean_r2 = np.array([per_T_regressions(B)[0].mean() for B in B_grid])
    B_best = float(B_grid[np.argmax(mean_r2)])
    r2s, lam0_T, slope_T = per_T_regressions(B_best)

    Tu = np.asarray(usable, float)
    lam_poly = tuple(np.polyfit(Tu, lam0_T, 2))       # lambda0(T)
    A_poly = tuple(np.polyfit(Tu, -slope_T, 2))       # A(T) = -slope

    return dict(z=int(z), lam=lam_poly, A=A_poly, B=B_best,
                diagnostics=dict(mean_R2=float(np.max(mean_r2)),
                                 per_T_R2=dict(zip(Tu.tolist(),
                                                   r2s.tolist())),
                                 temperatures_used=Tu.tolist(),
                                 n_points=int(lam.size)))
