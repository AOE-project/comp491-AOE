"""
Interactive iterative test for Analyser Agent.

Usage:
    python test_interactive.py

This script runs the analyser_node in a loop, allowing you to:
1. See the questions the agent asks
2. Provide answers interactively
3. Watch the model structure build up over multiple turns
4. See confirmed/unconfirmed assumptions evolve
"""

from agents.analyser import analyser_node


def base_state():
    """Create a fresh GraphState for testing."""
    return {
        "session_id": "interactive-test",
        "problem_description": "",
        "history": [],
        "iteration_count": 0,
        "analyser_output": None,
        "milp_model": {},
        "confirmed_assumptions": [],
        "unconfirmed_assumptions": [],
        "open_questions": [],
        "analysis_summary": "",
        "technical_summary": "",
        "analyser_approved": False,
        "raw_data": {},
        "generated_code": "",
        "solver_result": {},
        "explanation": "",
        "token_usage": {},
    }


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(result):
    """Pretty-print the agent's response."""
    print_section("AGENT RESPONSE")
    
    # Iteration count
    print(f"\n📊 Iteration: {result.get('iteration_count', 0)}")
    
    # Open questions
    questions = result.get("open_questions", [])
    if questions:
        print_section("OPEN QUESTIONS")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}\n")
    else:
        print_section("✅ NO MORE QUESTIONS - MODEL READY FOR APPROVAL")
    
    # Confirmed assumptions
    confirmed = result.get("confirmed_assumptions", [])
    if confirmed:
        print_section("CONFIRMED ASSUMPTIONS")
        for assumption in confirmed:
            print(f"  ✓ {assumption}")
    
    # Unconfirmed assumptions
    unconfirmed = result.get("unconfirmed_assumptions", [])
    if unconfirmed:
        print_section("UNCONFIRMED ASSUMPTIONS (Need Clarification)")
        for assumption in unconfirmed:
            print(f"  ? {assumption}")
    
    # Analysis summary
    summary = result.get("analysis_summary", "")
    if summary:
        print_section("ANALYSIS SUMMARY (Plain English)")
        print(summary)
    
    # Technical summary
    tech_summary = result.get("technical_summary", "")
    if tech_summary:
        print_section("TECHNICAL SUMMARY (Symbolic)")
        print(tech_summary)
    
    # MILP Model structure
    milp = result.get("milp_model", {})
    if milp:
        print_section("MILP MODEL STRUCTURE")
        for key in ["sets", "parameters", "variables", "objective", "constraints"]:
            if key in milp:
                val = milp[key]
                if isinstance(val, dict):
                    print(f"  {key}: {list(val.keys())}")
                else:
                    print(f"  {key}: {val}")


def run_interactive():
    """Main interactive loop."""
    state = base_state()
    turn = 0
    
    print_section("ANALYSER AGENT - INTERACTIVE TEST")
    print("\nThis tool lets you test the Analyser Agent interactively.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Initial problem description
    print("📝 First, provide the initial problem description:")
    print("   (Example: 'I want to minimize shipping cost from 3 warehouses to 4 stores')\n")
    
    problem = input("Problem: ").strip()
    if problem.lower() in ["quit", "exit"]:
        print("Exiting.")
        return
    
    state["problem_description"] = problem
    state["history"].append({"role": "user", "content": problem})
    
    # Main loop
    while True:
        turn += 1
        print_section(f"TURN {turn} - Running Analyser Node")
        
        try:
            result = analyser_node(state)
        except Exception as e:
            print(f"\n❌ ERROR during agent execution:\n{e}")
            break
        
        # Update state with result
        state.update(result)
        
        # Print result
        print_result(result)
        
        # Check if done
        open_questions = result.get("open_questions", [])
        if not open_questions or result.get("iteration_count", 0) >= 10:
            print_section("✅ MODEL STRUCTURE COMPLETE")
            if result.get("iteration_count", 0) >= 10:
                print("(Maximum iterations reached.)")
            else:
                print("All clarifying questions have been answered.")
                print("The model is ready for user approval.")
            break
        
        # Prompt for user response
        print_section("YOUR TURN - Answer the above questions")
        print("\nProvide your answers/clarifications to move forward.")
        print("(You can address multiple questions in one response.)\n")
        
        user_response = input("Your answer: ").strip()
        
        if user_response.lower() in ["quit", "exit"]:
            print("Exiting.")
            break
        
        # Add to history
        state["history"].append({"role": "user", "content": user_response})
        print()


if __name__ == "__main__":
    run_interactive()
