"""pipeline package

Houses signal extraction, rule engine, evidence builder and timeline
reconstruction modules. Each submodule should expose small, testable functions
and avoid side effects on import.
"""
__all__ = ["signals", "rules", "evidence", "timeline"]
