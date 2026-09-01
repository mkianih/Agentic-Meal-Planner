"""Put the project root on sys.path so these scripts can be run directly.

Kept as a tiny shared module rather than repeated in every test, and resolved
relative to this file so the tests work from any working directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
