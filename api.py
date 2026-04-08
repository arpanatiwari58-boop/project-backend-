from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import core

app = Flask(__name__)
CORS(app)

# Initialize generated training files only once.
if not os.path.exists("data/training/scenarios.npy"):
    core.main()

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        land = float(data.get('land', 1.0))
        budget = float(data.get('budget', 900000.0))
        epsilon = float(data.get('epsilon', 20000.0))

        # Load generated scenarios and profit matrix
        scenarios = np.load("data/training/scenarios.npy")
        profit_matrix = core.compute_profit_matrix(scenarios)

        # Solve model
        x_opt, s_opt, lam_opt, obj_val, status = core.solve_wdro_farm_model(
            profit_matrix=profit_matrix,
            total_land=land,
            total_budget=budget,
            epsilon=epsilon
        )

        if x_opt is None or status not in ["optimal", "optimal_inaccurate"]:
            return jsonify({
                "status": status,
                "message": "No feasible solution found"
            }), 400

        # Prepare response
        allocation = {}
        for crop, area in zip(core.CROPS, x_opt):
            if area > 1e-5:
                allocation[crop] = round(float(area), 4)

        scenario_profits = profit_matrix @ x_opt

        return jsonify({
            "status": status,
            "allocation": allocation,
            "metrics": {
                "min_profit": round(float(scenario_profits.min()), 2),
                "mean_profit": round(float(scenario_profits.mean()), 2),
                "max_profit": round(float(scenario_profits.max()), 2),
                "objective_value": round(float(obj_val), 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-validation', methods=['GET'])
def validate():
    try:
        # Load original scenarios to use as base for deviation
        scenarios = np.load("data/training/scenarios.npy")
        
        # We will generate 200 new samples by deviating existing scenarios
        # or just generating new ones from the same distribution.
        # User requested 200 new scenario data.
        
        results = []
        
        # Set a fixed seed for consistency in validation data
        np.random.seed(999)
        
        # Option: Generate 2 sets of 100 scenarios with deviations to get 200 total
        for set_idx in range(1, 3):
            test_scen = scenarios.copy().astype(float)
            rain_shift = np.random.uniform(-0.12, 0.12)
            yield_shifts = np.random.normal(0, 0.06, len(core.CROPS))
            price_shifts = np.random.normal(0, 0.05, len(core.CROPS))

            test_scen[:, 0] *= (1 + rain_shift)
            test_scen[:, 1:1 + len(core.CROPS)] *= (1 + yield_shifts)
            test_scen[:, 1 + len(core.CROPS):1 + 2*len(core.CROPS)] *= (1 + price_shifts)

            # Clip
            test_scen[:, 0] = np.clip(test_scen[:, 0], 500, 1800)
            test_scen[:, 1:1 + len(core.CROPS)] = np.clip(test_scen[:, 1:1 + len(core.CROPS)], 1.0, 70.0)
            test_scen[:, 1 + len(core.CROPS):1 + 2*len(core.CROPS)] = np.clip(
                test_scen[:, 1 + len(core.CROPS):1 + 2*len(core.CROPS)], 1000, 20000)

            # Convert to list of dicts for JSON
            for row in test_scen:
                scenario_data = {
                    "rainfall_mm": round(float(row[0]), 2),
                    "yields": {},
                    "prices": {}
                }
                for i, crop in enumerate(core.CROPS):
                    scenario_data["yields"][crop] = round(float(row[1 + i]), 4)
                    scenario_data["prices"][crop] = round(float(row[1 + len(core.CROPS) + i]), 2)
                
                results.append(scenario_data)

        return jsonify({
            "total_scenarios": len(results),
            "scenarios": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
