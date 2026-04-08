import numpy as np
import pandas as pd
import cvxpy as cp
import os
import pickle
import ecos

# =========================================================
# 1. BASIC SETUP
# =========================================================

CROPS = [
    "LR Rice (Sub1)",
    "HR Rice (Basmati)",
    "Maize",
    "Soybean",
    "Kodo Millet",
    "Black Gram (Urad)",
    "Moong Dal"
]

N_CROPS = len(CROPS)

# Cost in Rs/hectare
import numpy as np

COST = np.array([58000, 64000, 42000, 38000, 22000, 28000, 29000], dtype=float)

# User-defined farm resources
TOTAL_LAND = 1.0          # hectares
TOTAL_BUDGET = 900000.0   # Rs   ← note: you changed from 70000 to 700000

# =========================================================
# 2. HISTORICAL DATA
# =========================================================

YEARS = list(range(2014, 2024))

RAINFALL_MM = np.array([
    870, 1050, 1120, 1380, 1090, 940, 1290, 1110, 1020, 1060
], dtype=float)

YIELD_DATA = np.array([
    [31.0,  9.0, 24.0, 10.0, 19.0,  7.5,  6.0],
    [36.0, 30.0, 30.0, 15.0, 14.0, 10.0,  9.0],
    [40.0, 38.0, 46.0, 17.0, 12.0, 11.5, 10.5],
    [43.0, 36.0, 14.0,  7.0,  4.0,  4.5,  3.0],
    [41.0, 39.0, 44.0, 17.5, 13.0, 12.0, 11.0],
    [28.0,  7.0, 26.0, 12.0, 20.0,  8.0,  7.0],
    [44.0, 35.0, 12.0,  6.5,  3.5,  4.0,  2.5],
    [40.0, 37.0, 43.0, 16.5, 13.5, 11.0, 10.0],
    [35.0, 22.0, 34.0, 14.0, 16.0,  9.5,  8.5],
    [38.0, 34.0, 40.0, 16.0, 14.0, 10.5,  9.5],
], dtype=float)

PRICE_DATA = np.array([
    [2600, 4800, 2600, 5400, 4200,  9500, 10200],
    [2350, 3900, 2200, 4900, 3700,  8000,  8800],
    [2200, 3700, 2000, 4700, 3400,  7500,  8200],
    [2100, 3500, 2800, 5600, 5200,  9800, 11000],
    [2300, 3800, 2100, 4800, 3600,  7800,  8500],
    [2700, 5000, 2700, 5500, 4300,  9200, 10000],
    [2050, 3400, 2900, 5700, 5500, 10200, 11500],
    [2250, 3750, 2050, 4750, 3550,  7700,  8400],
    [2400, 4100, 2300, 5100, 3900,  8500,  9200],
    [2300, 3800, 2150, 4900, 3650,  7900,  8600],
], dtype=float)

# =========================================================
# 3. BUILD HISTORICAL DATAFRAME
# =========================================================

def clean_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")

def build_historical_dataframe():
    data = {
        "year": YEARS,
        "rainfall_mm": RAINFALL_MM
    }
    for i, crop in enumerate(CROPS):
        key = clean_name(crop)
        data[f"yield_{key}"] = YIELD_DATA[:, i]
        data[f"price_{key}"] = PRICE_DATA[:, i]
    return pd.DataFrame(data)

# =========================================================
# 4. MONTE CARLO SCENARIO GENERATOR
# =========================================================

def generate_monte_carlo_data(df, S=100, seed=42):
    np.random.seed(seed)

    feature_cols = ["rainfall_mm"]
    for crop in CROPS:
        key = clean_name(crop)
        feature_cols.append(f"yield_{key}")
    for crop in CROPS:
        key = clean_name(crop)
        feature_cols.append(f"price_{key}")

    X = df[feature_cols].values.astype(float)

    mu = X.mean(axis=0)
    Sigma = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])

    scenarios = np.random.multivariate_normal(mu, Sigma, size=S)

    # Clip realistic ranges
    scenarios[:, 0] = np.clip(scenarios[:, 0], 500, 1800)  # rainfall
    scenarios[:, 1:1 + N_CROPS] = np.clip(scenarios[:, 1:1 + N_CROPS], 1.0, 70.0)  # yields
    scenarios[:, 1 + N_CROPS:1 + 2 * N_CROPS] = np.clip(
        scenarios[:, 1 + N_CROPS:1 + 2 * N_CROPS], 1000, 20000)  # prices

    generated_df = pd.DataFrame(scenarios, columns=feature_cols)
    generated_df.insert(0, "scenario_id", np.arange(1, S + 1))

    return generated_df, scenarios

# =========================================================
# 5. PROFIT MATRIX
# =========================================================

def compute_profit_matrix(scenarios):
    yields = scenarios[:, 1:1 + N_CROPS]
    prices = scenarios[:, 1 + N_CROPS:1 + 2 * N_CROPS]
    profit_matrix = yields * prices - COST.reshape(1, -1)
    return profit_matrix

# =========================================================
# 6. WDRO MODEL SOLVER
# =========================================================

def solve_wdro_farm_model(profit_matrix, total_land, total_budget, epsilon=15.0):
    n_scenarios, n_crops = profit_matrix.shape

    x = cp.Variable(n_crops, nonneg=True)
    s = cp.Variable(n_scenarios)
    lam = cp.Variable(nonneg=True)

    objective = cp.Maximize(cp.sum(s) / n_scenarios - epsilon * lam)

    constraints = [
        cp.sum(x) <= total_land,
        COST @ x <= total_budget,
        cp.norm(x, 2) <= lam
    ]

    for j in range(n_scenarios):
        scenario_profit = profit_matrix[j, :] @ x
        constraints.append(s[j] <= scenario_profit)
        constraints.append(scenario_profit >= 0)  # no bankruptcy

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    return x.value, s.value, lam.value, prob.value, prob.status

# =========================================================
# 7. PRINT EPSILON-WISE REPORT
# =========================================================

def run_epsilon_experiment(profit_matrix, epsilon_values):
    for eps in epsilon_values:
        x_opt, s_opt, lam_opt, obj_val, status = solve_wdro_farm_model(
            profit_matrix=profit_matrix,
            total_land=TOTAL_LAND,
            total_budget=TOTAL_BUDGET,
            epsilon=eps
        )

        print("=" * 70)
        print(f"EPSILON = {eps}")

        if x_opt is None or status not in ["optimal", "optimal_inaccurate"]:
            print(f"Status   : {status}")
            print("No feasible solution found.\n")
            continue

        scenario_profit = profit_matrix @ x_opt

        print(f"Status   : {status}")
        print(f"Objective: {obj_val:,.2f}")
        print(f"Min Profit  : Rs {scenario_profit.min():,.2f}")
        print(f"Mean Profit : Rs {scenario_profit.mean():,.2f}")
        print(f"Max Profit  : Rs {scenario_profit.max():,.2f}")
        print("\nAllocation:")

        used_any = False
        for crop, area in zip(CROPS, x_opt):
            if area > 1e-5:
                used_any = True
                print(f"  {crop:<22} : {area:.4f} ha")

        if not used_any:
            print("  No crop selected.")
        print()

# =========================================================
# 8. MAIN — run optimization + save everything
# =========================================================

def main():
    # Create folders
    os.makedirs("data/historical", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("results/solutions", exist_ok=True)

    # Build historical data
    historical_df = build_historical_dataframe()

    # Generate Monte-Carlo scenarios
    generated_df, scenarios = generate_monte_carlo_data(
        historical_df, S=100, seed=42
    )

    # Compute profit matrix
    profit_matrix = compute_profit_matrix(scenarios)

    # Epsilon values to test
    epsilon_values = [20000
    ]

    # Store solutions
    solutions = {}

    print("Solving for different ε values...\n")
    for eps in epsilon_values:
        x_opt, s_opt, lam_opt, obj_val, status = solve_wdro_farm_model(
            profit_matrix=profit_matrix,
            total_land=TOTAL_LAND,
            total_budget=TOTAL_BUDGET,
            epsilon=eps
        )

        if x_opt is None or status not in ["optimal", "optimal_inaccurate"]:
            solutions[eps] = {
                "status": status,
                "allocation": None,
                "promised_min": None,
                "promised_mean": None,
                "promised_max": None,
                "objective": obj_val if obj_val is not None else None
            }
            print(f"Epsilon {eps:>8} → {status}")
            continue

        scenario_profit = profit_matrix @ x_opt

        solutions[eps] = {
            "status": status,
            "allocation": x_opt,
            "promised_min": scenario_profit.min(),
            "promised_mean": scenario_profit.mean(),
            "promised_max": scenario_profit.max(),
            "objective": obj_val
        }

        print(f"Epsilon {eps:>8} → solved | "
              f"Min: {scenario_profit.min():>8,.0f} | "
              f"Mean: {scenario_profit.mean():>8,.0f}")

    # Save everything
    historical_df.to_csv("data/historical/historical_data_2014_2023.csv", index=False)
    np.save("data/training/scenarios.npy", scenarios)
    generated_df.to_csv("data/training/generated_scenarios.csv", index=False)
    np.save("data/training/profit_matrix.npy", profit_matrix)

    with open("results/solutions/solutions_dict.pkl", "wb") as f:
        pickle.dump(solutions, f)

    print("\n" + "="*70)
    print("SAVED FILES:")
    print("  data/historical/historical_data_2014_2023.csv")
    print("  data/training/scenarios.npy")
    print("  data/training/generated_scenarios.csv")
    print("  data/training/profit_matrix.npy")
    print("  results/solutions/solutions_dict.pkl")
    print("="*70)

    # Optional: print full report
    print("\nFULL EPSILON-WISE REPORT")
    print("="*70)
    run_epsilon_experiment(profit_matrix, epsilon_values)

    return historical_df, generated_df, profit_matrix, scenarios, solutions


# ---- Execute ----
if __name__ == "__main__":
    historical_df, generated_df, profit_matrix, scenarios, solutions = main()