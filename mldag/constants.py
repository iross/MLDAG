"""Shared defaults used across mldag's DAG-generation and provenance modules.

Kept dependency-free (no htcondor2/typer imports) so any module can import
from here without pulling in daggen.py's runtime dependencies.
"""

DEFAULT_CLASSAD_FIELDS_FILE = "provenance_fields.yaml"
