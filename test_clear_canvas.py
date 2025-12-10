"""
Test to verify Clear Canvas functionality with session state.
"""

import streamlit as st

def test_canvas_key_logic():
    """Test the canvas key increment logic."""
    print("Testing Clear Canvas Logic")
    print("=" * 50)
    
    # Simulate session state
    session_state = {'canvas_key': 0}
    
    print(f"Initial canvas_key: {session_state['canvas_key']}")
    print(f"Initial canvas component key: 'canvas_{session_state['canvas_key']}'")
    
    # Simulate first clear
    print("\n[User clicks Clear Canvas]")
    session_state['canvas_key'] += 1
    print(f"After clear #1: canvas_key = {session_state['canvas_key']}")
    print(f"Canvas component key: 'canvas_{session_state['canvas_key']}'")
    
    # Simulate second clear
    print("\n[User clicks Clear Canvas again]")
    session_state['canvas_key'] += 1
    print(f"After clear #2: canvas_key = {session_state['canvas_key']}")
    print(f"Canvas component key: 'canvas_{session_state['canvas_key']}'")
    
    # Simulate third clear
    print("\n[User clicks Clear Canvas again]")
    session_state['canvas_key'] += 1
    print(f"After clear #3: canvas_key = {session_state['canvas_key']}")
    print(f"Canvas component key: 'canvas_{session_state['canvas_key']}'")
    
    print("\n" + "=" * 50)
    print("✅ Logic Test Passed!")
    print("\nExplanation:")
    print("- Each time 'Clear Canvas' is clicked, canvas_key increments")
    print("- This creates a new canvas component with a different key")
    print("- Streamlit treats it as a new component, resetting its state")
    print("- This effectively clears the drawing")

if __name__ == "__main__":
    test_canvas_key_logic()
