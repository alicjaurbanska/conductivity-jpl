
# =============================================================================
# validation tests
# =============================================================================
def _run_mccleskey_selftests(verbose=True):
    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # ---- 1. Infinite-dilution molal conductivities at 25 C vs literature ----
    # (limiting equivalent conductances * |z|, Robinson & Stokes 1965)
    lit = {"K+": 73.5, "Na+": 50.1, "H+": 349.8, "Cl-": 76.35,
           "Ca+2": 119.0, "Mg+2": 106.1, "SO4-2": 160.0, "OH-": 197.6}
    if verbose:
        print("Test 1: lambda0(25 C) vs literature limiting molal conductivities")
    for sp, ref in lit.items():
        lam0 = np.polyval(MCCLESKEY_2012_PARAMS[sp]["lam"], 25.0)
        err = abs(lam0 - ref) / ref
        check(f"{sp}: {lam0:.1f} vs {ref} ({100*err:.1f}%)", err < 0.02)

    # ---- 2. KCl solutions at 25 C vs primary-standard data (Pratt 2001) ----
    if verbose:
        print("Test 2: KCl conductivity vs molality-based primary standards")
    for m, kappa_ref_uScm, tol in [(0.01, 1408.2, 0.02), (0.1, 12852.0, 0.02),
                                   (1.0, 111310.0, 0.03)]:
        res = mccleskey2012_conductivity({"K+": m, "Cl-": m}, 25.0,
                                         validate_range=False)
        kap = float(res.sigma_mScm) * 1000.0  # uS/cm
        err = abs(kap - kappa_ref_uScm) / kappa_ref_uScm
        check(f"{m} mol/kg KCl: {kap:.0f} vs {kappa_ref_uScm:.0f} uS/cm "
              f"({100*err:.1f}%)", err < tol)

    # ---- 3. Ionic strength is computed from the FULL composition -----------
    if verbose:
        print("Test 3: ionic strength definition (Eq. 9) and bug regression")
    I = effective_ionic_strength({"Na+": 0.5, "Cl-": 0.5})
    check(f"0.5 mol/kg NaCl -> I = {float(I):.3f} (must be 0.500, "
          "not per-ion 0.250)", abs(float(I) - 0.5) < 1e-12)
    I2 = effective_ionic_strength({"Mg+2": 0.1, "Cl-": 0.2})
    check(f"0.1 mol/kg MgCl2 -> I = {float(I2):.3f} (must be 0.300)",
          abs(float(I2) - 0.3) < 1e-12)
    # Cross-ion coupling: adding an inert-for-conductivity electrolyte must
    # lower each lambda_i via the shared I (impossible with per-ion I).
    a = mccleskey2012_conductivity({"K+": 0.01, "Cl-": 0.01}, 25.0)
    b = mccleskey2012_conductivity({"K+": 0.01, "Cl-": 0.21, "Mg+2": 0.1},
                                   25.0)
    check("lambda_K in mixed solution < lambda_K in pure KCl "
          f"({float(b.lambda_i['K+']):.2f} < {float(a.lambda_i['K+']):.2f})",
          float(b.lambda_i["K+"]) < float(a.lambda_i["K+"]))

    # ---- 4. Double-weighting guard ------------------------------------------
    if verbose:
        print("Test 4: unrecognized species cannot silently corrupt I")
    try:
        effective_ionic_strength({"K+": 0.01, "Cl-": 0.01, "Weird+7": 0.01})
        check("unknown species raises ValueError", False)
    except ValueError:
        check("unknown species raises ValueError", True)

    # ---- 5. Neutral and parameterless charged species -----------------------
    if verbose:
        print("Test 5: ion pairs and neutral species handling")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = mccleskey2012_conductivity(
            {"Mg+2": 0.05, "SO4-2": 0.05, "MgSO4": 0.05, "MgCl+": 0.001,
             "Cl-": 0.001}, 25.0)
        warned = any("zero conductivity" in str(x.message) for x in w)
    check("MgSO4(aq) neutral: in input, zero I and zero sigma contribution",
          abs(float(res.I_eff_molkg) - (0.5*(0.05*4 + 0.05*4 + 0.001 + 0.001)))
          < 1e-12)
    check("charged pair without parameters (MgCl+) counted in I and warned",
          warned and "MgCl+" in res.ignored_species)

    # ---- 6. Naming conventions and CSV-column compatibility -----------------
    if verbose:
        print("Test 6: species-name normalization")
    same = [canonical_species_name(x) for x in
            ("K+", "K", "K_p1", "m_free_K")]
    check("K aliases -> 'K+'", all(s == "K+" for s in same))
    check("m_pair_KSO4 -> 'KSO4-'",
          canonical_species_name("m_pair_KSO4") == "KSO4-")
    check("SO4-- and SO4-2 -> 'SO4-2'",
          canonical_species_name("SO4--") == "SO4-2"
          and canonical_species_name("SO4-2") == "SO4-2")
    check("Fe+3 vs Fe defaults", canonical_species_name("Fe+3") == "Fe+3"
          and canonical_species_name("Fe+2") == "Fe+2")

    # ---- 7. Vectorization over T --------------------------------------------
    if verbose:
        print("Test 7: vectorization and physical T-dependence")
    T = np.array([0.0, 25.0, 50.0, 75.0, 95.0])
    res = mccleskey2012_conductivity({"Na+": 0.1, "Cl-": 0.1}, T)
    check("shape matches T grid", res.sigma_Sm.shape == T.shape)
    check("conductivity increases monotonically with T",
          bool(np.all(np.diff(res.sigma_Sm) > 0)))

    # ---- 8. Transport numbers ------------------------------------------------
    if verbose:
        print("Test 8: transport numbers (Eq. 6)")
    res = mccleskey2012_conductivity({"K+": 0.01, "Cl-": 0.01}, 25.0)
    tsum = sum(float(t) for t in res.transport_numbers.values())
    check(f"transport numbers sum to 1 ({tsum:.6f})", abs(tsum - 1) < 1e-12)

    # ---- 9. EA-1 Eq. (EA2): exact self-consistency ---------------------------
    if verbose:
        print("Test 9: Eq. EA2 recovers lambda of a known ion pair exactly")
    T = 25.0
    mix = {"K+": 0.015, "SO4-2": 0.008, "KSO4-": 0.002}
    full = mccleskey2012_conductivity(mix, T)
    known = {k: v for k, v in mix.items() if k != "KSO4-"}
    lam_rec, I_used = lambda_from_mixture(full.sigma_mScm, "KSO4-",
                                          mix["KSO4-"], known, T,
                                          target_z=-1)
    check(f"recovered lambda_KSO4 = {float(lam_rec):.4f} vs direct "
          f"{float(full.lambda_i['KSO4-']):.4f}",
          abs(float(lam_rec) - float(full.lambda_i["KSO4-"])) < 1e-10)
    check("I used in EA2 equals full-solution I",
          abs(float(I_used) - float(full.I_eff_molkg)) < 1e-15)

    # ---- 10. EA-1 fitting algorithm: parameter round-trip --------------------
    if verbose:
        print("Test 10: EA-1 B-optimized fit recovers Table-1 parameters")
    rng = np.random.default_rng(0)
    p_true = MCCLESKEY_2012_PARAMS["KSO4-"]
    Ts = np.array([5., 10., 25., 35., 45., 70., 90.])
    Is = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0])
    Tg, Ig = np.meshgrid(Ts, Is, indexing="ij")
    x = np.sqrt(Ig) / (1 + p_true["B"] * np.sqrt(Ig))
    lam_syn = np.polyval(p_true["lam"], Tg) - np.polyval(p_true["A"], Tg) * x
    fit = fit_mccleskey_parameters(Tg.ravel(), Ig.ravel(), lam_syn.ravel(),
                                   z=-1)
    xf = np.sqrt(Ig) / (1 + fit["B"] * np.sqrt(Ig))
    lam_fit = np.polyval(fit["lam"], Tg) - np.polyval(fit["A"], Tg) * xf
    maxerr = float(np.max(np.abs(lam_fit - lam_syn) / np.abs(lam_syn)))
    check(f"B recovered: {fit['B']:.2f} vs {p_true['B']} ; max lambda "
          f"error {100*maxerr:.3f}%", abs(fit["B"] - p_true["B"]) < 0.11
          and maxerr < 0.005)

    # ---- 11. Custom charged species allowed, neutral species rejected --------
    if verbose:
        print("Test 11: extra_params restricted to charged species")
    mgcl_eff = dict(z=+1, lam=(0.0, 0.5, 30.0), A=(0.0, 0.0, 0.0), B=1.0)
    base = {"Mg+2": 0.05, "Cl-": 0.11, "MgCl+": 0.01}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = mccleskey2012_conductivity(base, 25.0)  # MgCl+ inert (warned)
    r1 = mccleskey2012_conductivity(base, 25.0,
                                    extra_params={"MgCl+": mgcl_eff})
    dsig = float(r1.sigma_mScm - r0.sigma_mScm)
    expected = (0.5 * 25.0 + 30.0) * 0.01
    check(f"charged pair MgCl+ adds lambda*m = {dsig:.4f} mS/cm "
          f"(expected {expected:.4f})", abs(dsig - expected) < 1e-10)
    check("I identical with and without parameters (charge already counted)",
          abs(float(r1.I_eff_molkg) - float(r0.I_eff_molkg)) < 1e-15)
    try:
        mccleskey2012_conductivity(
            {"Mg+2": 0.05, "SO4-2": 0.05, "MgSO4": 0.05}, 25.0,
            extra_params={"MgSO4": dict(z=0, lam=(0, 0.5, 20.0),
                                        A=(0, 0, 0), B=1.0)})
        check("neutral species in extra_params raises ValueError", False)
    except ValueError:
        check("neutral species in extra_params raises ValueError", True)
    try:
        fit_mccleskey_parameters([5, 25, 45] * 3, [0.01] * 9, [1.0] * 9, z=0)
        check("fit with z = 0 raises ValueError", False)
    except ValueError:
        check("fit with z = 0 raises ValueError", True)

    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


_run_mccleskey_selftests()
