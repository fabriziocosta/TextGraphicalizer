"""Laya-backed ontology-constrained text graph extraction."""

from .aesop import (
    AESOP_SOURCE_URL,
    DEFAULT_AESOP_CACHE_PATH,
    load_aesop_fables,
    parse_aesop_fables,
)
from .errors import (
    GraphOptimizationError,
    LayaBackendError,
    LayaInferenceError,
    LayaModelError,
    LayaResponseError,
    OntologyError,
)
from .llm_backend import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_LLM_MODEL,
    DEFAULT_OPENAI_LLM_MODEL,
    OllamaGroundingBackend,
    OpenAIGroundingBackend,
)
from .nli_backend import NliGroundingBackend
from .ontology import Concept, Ontology, Relation, load_ontology
from .span_backend import ConceptDescription, Span, SpanGroundingBackend, SpanScore
from .stopwords import load_stopwords
from .transformer import TextGraphicalizer

__all__ = [
    "AESOP_SOURCE_URL",
    "Concept",
    "DEFAULT_AESOP_CACHE_PATH",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_LLM_MODEL",
    "DEFAULT_OPENAI_LLM_MODEL",
    "GraphOptimizationError",
    "LayaBackendError",
    "LayaInferenceError",
    "LayaModelError",
    "LayaResponseError",
    "Ontology",
    "OntologyError",
    "Relation",
    "NliGroundingBackend",
    "OllamaGroundingBackend",
    "OpenAIGroundingBackend",
    "ConceptDescription",
    "Span",
    "SpanGroundingBackend",
    "SpanScore",
    "TextGraphicalizer",
    "load_aesop_fables",
    "load_ontology",
    "load_stopwords",
    "parse_aesop_fables",
]

__version__ = "0.1.0"
