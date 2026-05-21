# -*- coding: utf-8 -*-
"""Compute exact optimization score for selected shops using Gurobi."""
import sys
import json
import pathlib
import gurobipy as gp
from gurobipy import GRB


def load_data():
    """Load data from JSON files in src/data directory."""
    data_dir = pathlib.Path(__file__).parent / "src" / "data"

    demand = json.loads((data_dir / "demand.json").read_text())
    capacity = json.loads((data_dir / "capacity.json").read_text())
    fixed_cost_daily = json.loads((data_dir / "fixed_cost_daily.json").read_text())
    visit_prob = json.loads((data_dir / "visit_prob.json").read_text())

    B = list(demand.keys())
    L = B.copy()

    return B, L, demand, capacity, fixed_cost_daily, visit_prob


def compute_score(selected_shops):
    """
    Compute the exact objective value for a given set of open shops.

    Args:
        selected_shops: list of shop names to open

    Returns:
        objective value (float)
    """
    B, L, demand, capacity, fixed_cost_daily, visit_prob = load_data()
    rev_per_customer = 150.0

    # Build Gurobi model with fixed open shops
    m = gp.Model("Donut_Shops_Score_Calc")
    m.Params.OutputFlag = 0

    # Decision variables
    open_var = m.addVars(L, vtype=GRB.BINARY, lb=0, ub=1, name="open")
    assign = m.addVars(B, L, vtype=GRB.BINARY, lb=0, ub=1, name="assign")
    served = m.addVars(B, L, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="served")

    # Fix open variables: set selected shops to 1, others to 0
    for shop in L:
        if shop in selected_shops:
            open_var[shop].lb = 1
            open_var[shop].ub = 1
        else:
            open_var[shop].lb = 0
            open_var[shop].ub = 0

    # Objective: maximize revenue - fixed costs
    m.setObjective(
        gp.quicksum(rev_per_customer * served[b, l] for b in B for l in L)
        - gp.quicksum(fixed_cost_daily[l] * open_var[l] for l in L),
        GRB.MAXIMIZE,
    )

    # Constraints
    # assignment_at_most_one_shop: for all b in B: sum_{l in L} assign[b,l] <= 1
    m.addConstrs(
        (gp.quicksum(assign[b, l] for l in L) <= 1 for b in B),
        name="assignment_at_most_one_shop",
    )

    # assigned_only_if_open: for all b in B, l in L: assign[b,l] <= open[l]
    m.addConstrs(
        (assign[b, l] <= open_var[l] for b in B for l in L),
        name="assigned_only_if_open",
    )

    # served_upper_link_when_assigned
    m.addConstrs(
        (
            served[b, l] <= (demand[b] * visit_prob[b][l]) * assign[b, l]
            for b in B
            for l in L
        ),
        name="served_upper_link_when_assigned",
    )

    # served_upper_by_pair_expectation
    m.addConstrs(
        (
            served[b, l] <= demand[b] * visit_prob[b][l]
            for b in B
            for l in L
        ),
        name="served_upper_by_pair_expectation",
    )

    # shop_capacity: for all l in L: sum_{b in B} served[b,l] <= capacity[l]
    m.addConstrs(
        (gp.quicksum(served[b, l] for b in B) <= capacity[l] for l in L),
        name="shop_capacity",
    )

    m.optimize()

    if m.status == GRB.OPTIMAL:
        return int(m.objVal)
    else:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_score.py shop1 shop2 shop3")
        sys.exit(1)

    shops = sys.argv[1:]
    score = compute_score(shops)

    if score is not None:
        print(json.dumps({"score": score, "status": "ok"}))
    else:
        print(json.dumps({"score": None, "status": "error"}))
