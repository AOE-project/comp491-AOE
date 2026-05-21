# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import pathlib
import json
import gurobipy as gp
from gurobipy import GRB

# Import your compute_score function from compute_score.py if separate
from compute_score import compute_score

app = Flask(__name__)
CORS(app)  # Enable CORS so React can call this API

@app.route('/api/compute-score', methods=['POST'])
def api_compute_score():
    data = request.get_json()
    if not data or 'shops' not in data:
        return jsonify({'score': None, 'status': 'error', 'message': 'Missing shops list'}), 400
    shops = data['shops']
    try:
        score = compute_score(shops)
        return jsonify({'score': score, 'status': 'ok'})
    except Exception as e:
        return jsonify({'score': None, 'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)