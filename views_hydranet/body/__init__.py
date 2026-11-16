"""The regression body's declared vocabulary (ADR-079).

`body` is the one config field that names what trains the regression head, what activation it
applies, and what the model emits from. `BODIES` is the closed set of values; `resolve_body` is the
seam every reader goes through. `legacy_body_spec` is the quarantined pre-ADR-079 rule, kept only
for saved artifacts and schema-bypassing drivers.

Torch-free at import, so `config_initializer` can depend on it.
"""

from views_hydranet.body.legacy import legacy_body_spec
from views_hydranet.body.vocabulary import BODIES, BodySpec, declared_bodies, resolve_body

__all__ = ["BODIES", "BodySpec", "declared_bodies", "legacy_body_spec", "resolve_body"]
