"""Pytest configuration and test environment setup."""

import streamlit.components.v2
from streamlit.components.v2.component_manager import BidiComponentManager

# Streamlit 1.54+ requires components to be discovered when imported outside active runtime
_manager = BidiComponentManager()
_manager.discover_and_register_components(start_file_watching=False)
streamlit.components.v2.get_bidi_component_manager = lambda: _manager
