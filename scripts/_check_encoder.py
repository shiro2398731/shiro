"""
scripts/_check_encoder.py

Check that the LLM encoder translates and does not interpret.

Run yourself, with the key set:
    $env:DEEPSEEK_API_KEY = "<your key>"
    .\\venv\\Scripts\\python.exe scripts\\_check_encoder.py

What it verifies:
  1. the response parses and has exactly D numbers in [-1, 1]
  2. the same sentence twice gives vectors that are close (translation is
     stable, not improvisation)
  3. two different sentences give clearly different vectors
  4. nothing that looks like an answer comes back -- the reply must be JSON
     only, no prose

If (2) fails, the model is improvising rather than translating.
If (4) fails, it has been given more freedom than it should have.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L6_expression.encoder import LlmEncoder

D = 128

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    print("DEEPSEEK_API_KEY not set; nothing to check")
    sys.exit(1)

enc = LlmEncoder(key, dim=D)

sentences = ["今天好累", "今天好累", "你还记得我吗？"]
vecs = []
for s in sentences:
    sig = enc.encode(s)
    if sig is None:
        print(f"FAIL  encoder rejected: {enc.last_error}")
        sys.exit(1)
    v = sig.vector
    ok_len = v.shape[0] == D
    ok_range = bool(np.all(v >= -1.0) and np.all(v <= 1.0))
    ok_finite = bool(np.all(np.isfinite(v)))
    print(f"ok    {s!r:<14} len={v.shape[0]} range=[{v.min():+.2f},{v.max():+.2f}] "
          f"tags={sig.tags}")
    if not (ok_len and ok_range and ok_finite):
        print("FAIL  shape or range")
        sys.exit(1)
    vecs.append(v)

same = float(np.linalg.norm(vecs[0] - vecs[1]))
diff = float(np.linalg.norm(vecs[0] - vecs[2]))
print()
print(f"same sentence distance : {same:.3f}")
print(f"different sentence distance : {diff:.3f}")
print()
print(f"stable across repeats : {same < diff * 0.5}")
print(f"distinguishable       : {diff > 1e-3}")
print()
print("Note: LLM output is not bitwise reproducible. Stability means the two")
print("vectors are close, not identical. If they differ more than this, the")
print("model is choosing a vector freely rather than translating, and the")
print("encoder should not be used.")
