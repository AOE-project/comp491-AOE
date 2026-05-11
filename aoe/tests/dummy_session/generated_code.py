import gurobipy as gp
from gurobipy import GRB

# Transportation problem — dummy fixture data
warehouses = ["I1", "I2"]
stores     = ["J1", "J2", "J3"]

supply = {"I1": 100, "I2": 200}
demand = {"J1":  80, "J2": 120, "J3":  80}
cost   = {
    "I1": {"J1": 2.0, "J2": 3.0, "J3": 4.0},
    "I2": {"J1": 3.0, "J2": 2.0, "J3": 1.0},
}

m = gp.Model("transportation")
m.Params.OutputFlag = 0

x = m.addVars(warehouses, stores, lb=0, name="x")

m.setObjective(
    gp.quicksum(cost[i][j] * x[i, j] for i in warehouses for j in stores),
    GRB.MINIMIZE,
)

m.addConstrs(
    (gp.quicksum(x[i, j] for j in stores) <= supply[i] for i in warehouses),
    name="supply",
)
m.addConstrs(
    (gp.quicksum(x[i, j] for i in warehouses) >= demand[j] for j in stores),
    name="demand",
)

m.optimize()

if m.Status == GRB.OPTIMAL:
    print(f"Objective value: {m.ObjVal:.4f}")
    print("\nOptimal shipments:")
    for i in warehouses:
        for j in stores:
            v = x[i, j].X
            if v > 1e-6:
                print(f"  {i} -> {j}: {v:.2f}")
else:
    print(f"Optimization ended with status {m.Status}")
