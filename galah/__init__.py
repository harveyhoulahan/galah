import sys

# Windows consoles default to cp1252; reports use unicode dots and arrows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
