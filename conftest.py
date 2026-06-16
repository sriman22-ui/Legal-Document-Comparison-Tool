"""Makes the project root importable during tests.

pytest adds the directory containing the top-level conftest.py to sys.path, so
`from src.alignment import align` resolves when running `pytest` from the project root.
"""
