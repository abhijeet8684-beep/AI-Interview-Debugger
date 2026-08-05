AI-Interview-Debugger
=====================

Purpose
-------
AI-powered Interview Failure Debugger: analyze failed AI interview sessions and determine the most likely root cause.

Pipeline (high-level)
----------------------
1. Synthetic Interview Sessions
2. Signal Extraction
3. Rule Engine
4. Evidence Builder
5. Similarity Search (Sentence Transformers + FAISS)
6. LLM Reasoning Layer
7. Confidence Scoring
8. Streamlit Dashboard

Project layout
--------------
- app/            : Streamlit dashboard and app entrypoint
- data/           : Raw and processed synthetic sessions
- pipeline/       : Signal extraction, rule engine, evidence builder
- models/         : ML models, embeddings, FAISS index
- utils/          : Helpers, IO, logging
- docs/           : Design notes and developer docs
- tests/          : Unit and integration tests

Getting started
---------------
1. Create a virtual environment: python -m venv .venv
2. Activate it and install dependencies: pip install -r requirements.txt
3. Implement modules incrementally. See docs/ for module specs.

Development approach
--------------------
Work incrementally. Implement one module at a time and run tests. Keep functions small, typed, and documented.

Next step
---------
Implement module 1: project structure (done). Confirm and approve to continue with Module 2: dataset schema.
