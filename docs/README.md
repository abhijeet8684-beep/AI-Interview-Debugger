Documentation
=============

This docs folder holds design notes, module API docs, and decisions made during
implementation. For each module implemented, add a short design doc describing:
- purpose and inputs/outputs
- edge cases and assumptions
- test plan

Workflow
-------
1. Implement module in code
2. Add docs/<module>.md describing behavior and public API
3. Add tests in tests/test_<module>.py
4. Run targeted tests with pytest
