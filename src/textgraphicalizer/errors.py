class OntologyError(ValueError):
    """Raised when an ontology cannot be loaded or fails validation."""


class GraphOptimizationError(RuntimeError):
    """Raised when the requested graph constraints have no feasible solution."""
