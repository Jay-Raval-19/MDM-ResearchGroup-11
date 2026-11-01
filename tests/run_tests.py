"""Minimal test runner so the suite works without pytest installed."""
import importlib.util
import sys
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
failed = 0
total = 0
for path in sorted(here.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        total += 1
        try:
            getattr(module, name)()
            print(f"  pass  {path.stem}.{name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {path.stem}.{name}")
            traceback.print_exc()
print(f"\n{total - failed}/{total} tests passed")
sys.exit(1 if failed else 0)
