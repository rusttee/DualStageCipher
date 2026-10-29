"""Start the complete offline browser GUI.

Run:
    python extended_main.py

The browser page provides operations 0-5, full-media processing, paper-stage
tracing, matrix/bit tables, and histogram analysis.  No Matplotlib/Tk windows
are opened.
"""
from gui_server import launch

if __name__ == "__main__":
    launch()
