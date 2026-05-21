# Score Computation API

This directory contains a Flask-based API that computes exact Gurobi-optimized scores for selected donut shop configurations.

## How It Works

When you select 3 buildings in the web app, instead of using a greedy JavaScript approximation, the app now:

1. Calls the Python backend API via `POST /api/compute-score`
2. The API runs Gurobi with the selected shops fixed as open
3. Gurobi optimizes the customer assignments and allocation
4. Returns the exact objective value (revenue - fixed costs)

## Setup & Running

### Prerequisites
- Python 3.8+
- Gurobi (with valid license)
- Flask: `pip install flask`

### Quick Start

**Terminal 1 - Start the API server:**
```bash
cd Koc_use_case
chmod +x start_api.sh
./start_api.sh
```

Or directly:
```bash
python3 api.py
```

The API will run on `http://localhost:5000`

**Terminal 2 - Start the React app:**
```bash
cd Koc_use_case
npm run dev
```

## Files

- `compute_score.py` - Core optimization logic using Gurobi
- `api.py` - Flask API server
- `start_api.sh` - Startup script
- `CampusMap.jsx` - Frontend component that calls the API

## API Endpoint

**POST** `/api/compute-score`

Request body:
```json
{
  "shops": ["Shop1", "Shop2", "Shop3"]
}
```

Response (success):
```json
{
  "score": 45000,
  "status": "ok"
}
```

Response (error):
```json
{
  "error": "Must select exactly 3 shops"
}
```

## How Scoring Works

The Gurobi model:
- Fixes the 3 selected shops as open
- Optimizes customer assignments to shops
- Respects shop capacity constraints
- Maximizes: (revenue from served customers) - (daily fixed costs)

Formula:
```
Objective = Σ(150 × served[building,shop]) - Σ(fixed_cost[shop] × open[shop])
```

Where:
- `served[b,l]` = actual customers from building b served at shop l (continuous)
- `fixed_cost[l]` = daily operating cost for shop l
- Only 3 shops can be open
- Each building assigned to at most 1 shop
- Shop capacity constraints are respected

## Troubleshooting

**"Cannot connect to API"**
- Make sure `api.py` is running on port 5000
- Check for port conflicts: `lsof -i :5000`

**"Gurobi license not found"**
- Install Gurobi and obtain a valid license
- Set up Gurobi environment: `grbgetkey [key]`

**"Optimization failed"**
- Check the Flask console for detailed error messages
- The selected configuration may be infeasible
