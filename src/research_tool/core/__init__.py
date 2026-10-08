"""Framework core: runs, artifacts, state, pipeline resolution and the engine.

Nothing in ``core`` knows about any specific agent; agents plug in through the
registry and talk to each other only through artifacts.
"""
