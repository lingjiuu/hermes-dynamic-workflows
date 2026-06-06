"""Host port — the anti-corruption layer between this plugin and Hermes.

This is the only package allowed to import Hermes internals (``hermes_state``,
``hermes_constants``, ``gateway.*`` here; ``run_agent``, ``hermes_cli.*``,
``tools.*``, ``model_tools`` are slated to follow). Everything else imports
these thin, behavior-preserving wrappers, so a Hermes rename touches exactly
one place instead of being hunted across the tree.

These paths talk to a live Hermes and are exercised by a real run, not the test
suite (which injects fakes at the runner/manager boundary). Keep each wrapper a
faithful, byte-for-byte relocation of the call it replaced.
"""
