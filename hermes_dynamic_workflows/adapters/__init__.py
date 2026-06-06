"""Hermes-facing adapters: the composition edge of the plugin.

Tool handlers (``workflow``, ``task_stop``), the ``/workflows`` slash command,
and the ``pre_tool_call`` approval hook registered with Hermes by ``entry.py``.
This is the top layer; it may depend on any other package.
"""
