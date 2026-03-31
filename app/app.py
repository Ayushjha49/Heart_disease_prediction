import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify



ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict
from src.utils.data_preprocessing import encode_input, FEATURE_COLUMNS
from src.evaluate_params import train_and_evaluate_model, evaluate_parameter_range


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
	try:
		form_data = {name: request.form[name] for name in FEATURE_COLUMNS}
		input_data = encode_input(form_data)
		result = predict(input_data)
		return jsonify({"result": result})
	except KeyError as e:
		app.logger.error(f"KeyError in form submission: {e}")
		return jsonify({"error": f"Missing or invalid field: {e}. Please fill all fields correctly."}), 400
	except ValueError as e:
		app.logger.error(f"ValueError in form submission: {e}")
		return jsonify({"error": "Invalid input. Please enter valid values in all fields."}), 400
	except Exception as e:
		app.logger.error(f"Unexpected error: {e}")
		return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/evaluate", methods=["POST"])
def evaluate_route():
	"""
	Evaluate model with custom parameters
	Expects: n_trees, max_depth, min_samples_split
	"""
	try:
		data = request.get_json()
		n_trees = float(data.get("n_trees", 100))
		max_depth = float(data.get("max_depth", 5))
		min_samples_split = float(data.get("min_samples_split", 5))

		result = train_and_evaluate_model(n_trees, max_depth, min_samples_split)
		return jsonify(result)
	except Exception as e:
		app.logger.error(f"Error in evaluation: {e}")
		return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


@app.route("/evaluate-param-range", methods=["POST"])
def evaluate_param_range_route():
	"""
	Evaluate model across a range of parameter values
	Expects: param_name, param_range, base_params
	"""
	try:
		data = request.get_json()
		param_name = data.get("param_name")
		param_range = data.get("param_range", [])
		base_params = data.get("base_params", {})

		result = evaluate_parameter_range(param_name, param_range, base_params)
		return jsonify(result)
	except Exception as e:
		app.logger.error(f"Error in parameter range evaluation: {e}")
		return jsonify({"error": f"Parameter evaluation failed: {str(e)}"}), 500


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=8080)
