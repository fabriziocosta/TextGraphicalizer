class OntologyError(ValueError):
    """Raised when an ontology cannot be loaded or fails validation."""


class GraphOptimizationError(RuntimeError):
    """Raised when the requested graph constraints have no feasible solution."""


class LayaBackendError(RuntimeError):
    """Base class for model loading, inference, and response failures."""


class LayaModelError(LayaBackendError):
    """Raised when the Laya package or checkpoint cannot be loaded."""


class LayaInferenceError(LayaBackendError):
    """Raised when Laya fails while evaluating a batch of questions."""


class LayaResponseError(LayaBackendError, ValueError):
    """Raised when Laya returns a response with an invalid schema or value."""
