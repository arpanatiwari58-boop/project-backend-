from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

import core
import validation
import backtest_validation

app = Flask(__name__)
CORS(app)


def ensure_training_data():
    scenarios_path = "data/training/scenarios.npy"
    if not os.path.exists(scenarios_path):
        core.main()


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Backend is running",
        "endpoints": ["/health", "/solve", "/test-validation", "/evaluate-backtest"]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/solve', methods=['POST'])
def solve():
    try:
        ensure_training_data()
        data = request.json
        land = float(data.get('land', 1.0))
        budget = float(data.get('budget', 900000.0))
        epsilon = float(data.get('epsilon', 20000.0))

        scenarios = np.load("data/training/scenarios.npy")
        profit_matrix = core.compute_profit_matrix(scenarios)

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


@app.route('/test-validation', methods=['GET', 'POST'])
def validate():
    try:
        ensure_training_data()

        user_params = {}
        if request.method == 'POST' and request.is_json:
            user_params = request.json or {}

        structured_data = validation.get_validation_data(user_params)
        return jsonify(structured_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate-backtest', methods=['POST'])
def evaluate_backtest():
    try:
        ensure_training_data()
        data = request.json
        allocation = data.get('allocation', {})

        if not allocation:
            return jsonify({"error": "No allocation provided. Please pass an 'allocation' object."}), 400

        report = backtest_validation.run_backtest(allocation)
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
