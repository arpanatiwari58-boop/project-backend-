import numpy as np
import core


def format_scenarios_for_api(scenarios):
    """Converts a numpy scenario matrix into a clean list of dictionaries for JSON output."""
    results = []
    for row in scenarios:
        scenario_data = {
            "rainfall_mm": round(float(row[0]), 2),
            "yields": {},
            "prices": {}
        }
        for i, crop in enumerate(core.CROPS):
            scenario_data["yields"][crop] = round(float(row[1 + i]), 4)
            scenario_data["prices"][crop] = round(float(row[1 + core.N_CROPS + i]), 2)
        results.append(scenario_data)
    return results


def clip_scenarios(sc):
    """Ensures no mathematical shifts push constraints beyond realistic physical bounds."""
    sc[:, 0] = np.clip(sc[:, 0], 500, 1800)
    sc[:, 1:1 + core.N_CROPS] = np.clip(sc[:, 1:1 + core.N_CROPS], 1.0, 70.0)
    sc[:, 1 + core.N_CROPS:1 + 2 * core.N_CROPS] = np.clip(sc[:, 1 + core.N_CROPS:1 + 2 * core.N_CROPS], 1000, 20000)
    return sc


def get_validation_data(user_params):
    """
    Generates all 4 validation test sets (Baseline Shift, Volatility, Regime Shift, Black Swan).
    Reads 'user_params' dictionary if passed from the frontend for custom percentages, otherwise uses defaults.
    """
    base_scenarios = np.load("data/training/scenarios.npy").astype(float)
    S = len(base_scenarios)

    np.random.seed(999)

    rain_shift = float(user_params.get('rain_shift', -0.10))
    yield_shift = float(user_params.get('yield_shift', 0.05))
    price_shift = float(user_params.get('price_shift', -0.05))

    scen_baseline = base_scenarios.copy()
    scen_baseline[:, 0] *= (1 + rain_shift)
    scen_baseline[:, 1:1 + core.N_CROPS] *= (1 + yield_shift)
    scen_baseline[:, 1 + core.N_CROPS:] *= (1 + price_shift)
    scen_baseline = clip_scenarios(scen_baseline)

    volatility = float(user_params.get('volatility', 0.25))
    scen_vol = base_scenarios.copy()

    means = scen_vol.mean(axis=0)
    noise = np.random.normal(0, volatility * means, size=scen_vol.shape)
    scen_vol += noise
    scen_vol = clip_scenarios(scen_vol)

    p_d = float(user_params.get('p_drought', 0.50))
    p_n = float(user_params.get('p_normal', 0.30))
    p_f = float(user_params.get('p_flood', 0.20))

    def fit_regime(idx):
        X = np.column_stack([core.RAINFALL_MM[idx], core.YIELD_DATA[idx], core.PRICE_DATA[idx]])
        mu = X.mean(axis=0)
        if len(idx) > 1:
            Sigma = np.cov(X.T) + 1e-4 * np.eye(X.shape[1])
        else:
            Sigma = np.diag(np.abs(X[0]) * 0.10 + 1.0)
        return mu, Sigma

    mu_d, Sig_d = fit_regime(core.DROUGHT_IDX)
    mu_n, Sig_n = fit_regime(core.NORMAL_IDX)
    mu_f, Sig_f = fit_regime(core.FLOOD_IDX)

    n_d = int(S * p_d)
    n_n = int(S * p_n)
    n_f = S - n_d - n_n

    sc_d = np.random.multivariate_normal(mu_d, Sig_d, size=n_d) if n_d > 0 else np.empty((0, base_scenarios.shape[1]))
    sc_n = np.random.multivariate_normal(mu_n, Sig_n, size=n_n) if n_n > 0 else np.empty((0, base_scenarios.shape[1]))
    sc_f = np.random.multivariate_normal(mu_f, Sig_f, size=n_f) if n_f > 0 else np.empty((0, base_scenarios.shape[1]))

    scen_regime = np.vstack([x for x in [sc_d, sc_n, sc_f] if x.shape[0] > 0])
    scen_regime = clip_scenarios(scen_regime)

    swan_yield_drop = float(user_params.get('swan_yield_drop', -0.40))
    swan_price_drop = float(user_params.get('swan_price_drop', -0.30))

    scen_swan = base_scenarios.copy()
    scen_swan[:, 1:1 + core.N_CROPS] *= (1 + swan_yield_drop)
    scen_swan[:, 1 + core.N_CROPS:] *= (1 + swan_price_drop)
    means = scen_swan.mean(axis=0)
    noise_swan = np.random.normal(0, 0.15 * means, size=scen_swan.shape)
    scen_swan += noise_swan
    scen_swan = clip_scenarios(scen_swan)

    return {
        "baseline_shift": format_scenarios_for_api(scen_baseline),
        "volatility": format_scenarios_for_api(scen_vol),
        "regime_shift": format_scenarios_for_api(scen_regime),
        "black_swan": format_scenarios_for_api(scen_swan)
    }