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
    DEFAULT_MLX_LM_BASE_URL,
    DEFAULT_MLX_LM_CHAT_TEMPLATE_KWARGS,
    DEFAULT_MLX_LM_MAX_TOKENS,
    DEFAULT_MLX_LM_MODEL,
    DEFAULT_MLX_LM_MODEL_PATH,
    DEFAULT_MLX_LM_PYTHON,
    DEFAULT_MLX_LM_SERVER_HOST,
    DEFAULT_MLX_LM_SERVER_LOG_LEVEL,
    DEFAULT_MLX_LM_SERVER_PORT,
    DEFAULT_MLX_LM_TEMPERATURE,
    DEFAULT_MLX_LM_TIMEOUT,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_LLM_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_OPENAI_LLM_MODEL,
    MlxLmGroundingBackend,
    OllamaGroundingBackend,
    OpenAIGroundingBackend,
)
from .llm_config import (
    LLM_CONFIG_VERSION,
    SUPPORTED_LLM_PROVIDERS,
    LLMConfig,
    load_llm_config,
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
    "DEFAULT_MLX_LM_BASE_URL",
    "DEFAULT_MLX_LM_CHAT_TEMPLATE_KWARGS",
    "DEFAULT_MLX_LM_MAX_TOKENS",
    "DEFAULT_MLX_LM_MODEL",
    "DEFAULT_MLX_LM_MODEL_PATH",
    "DEFAULT_MLX_LM_PYTHON",
    "DEFAULT_MLX_LM_SERVER_HOST",
    "DEFAULT_MLX_LM_SERVER_LOG_LEVEL",
    "DEFAULT_MLX_LM_SERVER_PORT",
    "DEFAULT_MLX_LM_TEMPERATURE",
    "DEFAULT_MLX_LM_TIMEOUT",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_LLM_MODEL",
    "DEFAULT_OLLAMA_TIMEOUT",
    "DEFAULT_OPENAI_LLM_MODEL",
    "GraphOptimizationError",
    "LayaBackendError",
    "LayaInferenceError",
    "LayaModelError",
    "LayaResponseError",
    "LLMConfig",
    "LLM_CONFIG_VERSION",
    "Ontology",
    "OntologyError",
    "Relation",
    "NliGroundingBackend",
    "MlxLmGroundingBackend",
    "OllamaGroundingBackend",
    "OpenAIGroundingBackend",
    "ConceptDescription",
    "Span",
    "SpanGroundingBackend",
    "SpanScore",
    "TextGraphicalizer",
    "SUPPORTED_LLM_PROVIDERS",
    "load_aesop_fables",
    "load_ontology",
    "load_llm_config",
    "load_stopwords",
    "parse_aesop_fables",
]

__version__ = "0.1.0"
