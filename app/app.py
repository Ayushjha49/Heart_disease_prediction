import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify


# Ensure imports work when running: python app/app.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict
from src.utils.data_preprocessing import encode_input, FEATURE_COLUMNS


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


if __name__ == "__main__":
	app.run(debug=True)
