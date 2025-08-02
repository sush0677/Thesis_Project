#!/usr/bin/env python
"""
Simple runner for the MARL-GCP Simplified Dashboard
==================================================

This script runs the streamlined dashboard that focuses only on essential features
for thesis presentation.

Usage:
    python run_simplified_dashboard.py
"""

import streamlit as st
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplified_dashboard import SimplifiedMARLDashboard

def main():
    """Run the simplified dashboard."""
    try:
        dashboard = SimplifiedMARLDashboard()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"Error running dashboard: {e}")
        st.info("Please ensure all dependencies are installed and data files are available.")

if __name__ == "__main__":
    main() 