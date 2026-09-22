# Step 11: Three-dimensional receiver geometry

Place the files as follows:

- `geometry_step11.py` -> `fa_ddm/geometry.py`
- `test_geometry.py` -> `tests/test_geometry.py`

Run:

```powershell
python -m pytest tests\test_geometry.py -v
python -m pytest tests -v
```

Expected result for the new file: six tests pass.

This module converts Bob and Eve positions into distance and direction. It is
the basis for future vulnerability maps over Eve's 3-D position.
