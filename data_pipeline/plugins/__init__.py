"""Drop-in plugin directory for data_pipeline.

Any ``*.py`` file dropped in here is auto-imported at startup by
:func:`data_pipeline.plugin_loader.load_dropin_plugins`. Use the
``@register_fusion`` / ``@register_calibration`` decorators from
:mod:`data_pipeline.plugins_api` inside the file to register your plugin.

See ``docs/PLUGINS.md`` for the contract and a copy-paste skeleton, and
``example_passthrough.py`` in this directory for a minimal working example.
"""
