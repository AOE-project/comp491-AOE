"""
solver/runner.py — SolverRunner and SolverParser.

SolverRunner runs a Gurobi script in a subprocess and returns a SolverResult dict.
SolverParser extracts status, objective value, variables, and MIP gap from Gurobi's output.
Both raise SolverError on failure.
"""
