"""Versioned ontology loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .errors import OntologyError


def _as_id_list(value: Any, field: str, owner: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise OntologyError(f"{owner}.{field} must be a non-empty list when provided")
    values = tuple(str(item) for item in value)
    if any(not item for item in values):
        raise OntologyError(f"{owner}.{field} cannot contain empty IDs")
    if len(set(values)) != len(values):
        raise OntologyError(f"{owner}.{field} contains duplicate IDs")
    return values


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Relation:
    id: str
    label: str
    description: str
    source_concepts: tuple[str, ...] | None = None
    target_concepts: tuple[str, ...] | None = None

    def allows(self, source_id: str, target_id: str) -> bool:
        return (
            (self.source_concepts is None or source_id in self.source_concepts)
            and (self.target_concepts is None or target_id in self.target_concepts)
        )


@dataclass(frozen=True)
class Ontology:
    version: int
    concepts: tuple[Concept, ...]
    relations: tuple[Relation, ...]

    @property
    def concept_by_id(self) -> dict[str, Concept]:
        return {concept.id: concept for concept in self.concepts}

    @property
    def relation_by_id(self) -> dict[str, Relation]:
        return {relation.id: relation for relation in self.relations}

    def valid_relations(self, source_id: str, target_id: str) -> tuple[Relation, ...]:
        return tuple(
            relation
            for relation in self.relations
            if relation.allows(source_id, target_id)
        )


def _read_source(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if not path.exists():
        raise OntologyError(f"Ontology file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                data = json.load(handle)
            elif path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(handle)
            else:
                raise OntologyError("Ontology files must use .json, .yaml, or .yml")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise OntologyError(f"Could not read ontology {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise OntologyError("Ontology root must be a mapping")
    return data


def _required_text(item: Mapping[str, Any], field: str, owner: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OntologyError(f"{owner}.{field} must be a non-empty string")
    return value.strip()


def load_ontology(source: str | Path | Mapping[str, Any] | Ontology) -> Ontology:
    """Load and validate an ontology from a path, mapping, or Ontology object."""
    if isinstance(source, Ontology):
        return source
    data = _read_source(source)
    version = data.get("version")
    if version != 1:
        raise OntologyError("Only ontology schema version 1 is supported")

    raw_concepts = data.get("concepts")
    raw_relations = data.get("relations")
    if not isinstance(raw_concepts, Sequence) or isinstance(raw_concepts, (str, bytes)) or not raw_concepts:
        raise OntologyError("Ontology must contain a non-empty concepts list")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)) or not raw_relations:
        raise OntologyError("Ontology must contain a non-empty relations list")

    concepts: list[Concept] = []
    for index, raw in enumerate(raw_concepts):
        owner = f"concepts[{index}]"
        if not isinstance(raw, Mapping):
            raise OntologyError(f"{owner} must be a mapping")
        concept_id = _required_text(raw, "id", owner)
        concepts.append(
            Concept(
                id=concept_id,
                label=_required_text(raw, "label", owner),
                description=_required_text(raw, "description", owner),
            )
        )

    concept_ids = {concept.id for concept in concepts}
    if len(concept_ids) != len(concepts):
        raise OntologyError("Concept IDs must be unique")

    relations: list[Relation] = []
    for index, raw in enumerate(raw_relations):
        owner = f"relations[{index}]"
        if not isinstance(raw, Mapping):
            raise OntologyError(f"{owner} must be a mapping")
        relation = Relation(
            id=_required_text(raw, "id", owner),
            label=_required_text(raw, "label", owner),
            description=_required_text(raw, "description", owner),
            source_concepts=_as_id_list(raw.get("source_concepts"), "source_concepts", owner),
            target_concepts=_as_id_list(raw.get("target_concepts"), "target_concepts", owner),
        )
        for field, values in (
            ("source_concepts", relation.source_concepts),
            ("target_concepts", relation.target_concepts),
        ):
            if values is not None:
                unknown = set(values) - concept_ids
                if unknown:
                    raise OntologyError(
                        f"{owner}.{field} references unknown concepts: {sorted(unknown)}"
                    )
        relations.append(relation)

    relation_ids = {relation.id for relation in relations}
    if len(relation_ids) != len(relations):
        raise OntologyError("Relation IDs must be unique")

    return Ontology(version=1, concepts=tuple(concepts), relations=tuple(relations))
