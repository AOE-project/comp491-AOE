"""
middleware package — The only entry point to the AOE graph.

All external consumers (Gradio UI, CLI, tests) must call AOEHandle.run().
No external code can invoke compiled_graph directly.
"""
