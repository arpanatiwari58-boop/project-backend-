import numpy as np
import core


def run_backtest(allocation_dict):
    """
    Executes the 3-Layer Validation (Out-of-Sample, Historical, Regime-Stratified).
    Expects allocation_dict like {"LR Rice (Sub1)": 2.5, "Maize": 1.0, ...}
    """
    x_opt = np.zeros(core.N_CROPS)
    for i, crop in enumerate(core.CROPS):
        x_opt[i] = float(allocation_dict.get(crop, 0.0))

    hist_yields = core.YIELD_DATA
    hist_prices = core.PRICE_DATA
    hist_cost = core.COST

    train_scenarios = np.load("data/training/scenarios.npy").astype(float)
    profit_train = core.compute_profit_matrix(train_scenarios)

    hist_df = core.build_historical_dataframe()
    _, val_scenarios = core.generate_monte_carlo_data(hist_df, S=500, seed=99)
    profit_val = core.compute_profit_matrix(val_scenarios)

    train_profits = profit_train @ x_opt
    val_profits = profit_val @ x_opt

    layer1 = {
        "train_mean": float(train_profits.mean()),
        "train_min": float(train_profits.min()),
        "train_max": float(train_profits.max()),
        "train_std": float(train_profits.std()),
        "val_mean": float(val_profits.mean()),
        "val_min": float(val_profits.min()),
        "val_max": float(val_profits.max()),
        "val_std": float(val_profits.std()),
        "generalization_gap": float(train_profits.mean() - val_profits.mean())
    }

    historical_profit = (hist_yields * hist_prices - hist_cost) @ x_opt

    regime_labels = [
        "Drought", "Normal", "Normal", "Flood",
        "Normal", "Drought", "Flood", "Normal",
        "Normal", "Normal"
    ]

    layer2_list = []
    for i, year in enumerate(core.YEARS):
        layer2_list.append({
            "year": int(year),
            "regime": regime_labels[i],
            "profit": float(historical_profit[i])
        })

    layer2_summary = {
        "mean": float(historical_profit.mean()),
        "min": float(historical_profit.min()),
        "max": float(historical_profit.max()),
        "std": float(historical_profit.std())
    }

    rainfall = val_scenarios[:, 0]
    drought_mask = rainfall < 950
    flood_mask = rainfall > 1250
    normal_mask = ~drought_mask & ~flood_mask

    layer3 = {}
    for label, mask in [("Drought", drought_mask), ("Normal", normal_mask), ("Flood", flood_mask)]:
        n = int(mask.sum())
        if n == 0:
            layer3[label] = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        else:
            p = val_profits[mask]
            layer3[label] = {
                "count": n,
                "mean": float(p.mean()),
                "min": float(p.min()),
                "max": float(p.max())
            }

    return {
        "layer1_outofsample": layer1,
        "layer2_historical": {
            "years": layer2_list,
            "summary": layer2_summary
        },
        "layer3_regimes": layer3
    }