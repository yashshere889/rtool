"""The web UI: a JSON API over runs plus the built single-page app (``rtool ui``).

The server reads run directories directly and changes them only the way the
CLI does: by queuing actions for the next episode and waking one up. It never
calls the LLM. Import it only from the ``ui`` command: it needs the ``web``
extra (FastAPI, uvicorn).
"""
