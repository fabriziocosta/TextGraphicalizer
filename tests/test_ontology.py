import pytest

from textgraphicalizer import OntologyError, load_ontology


def valid_data():
    return {
        "version": 1,
        "concepts": [
            {"id": "a", "label": "A", "description": "Concept A."},
            {"id": "b", "label": "B", "description": "Concept B."},
        ],
        "relations": [
            {
                "id": "causes",
                "label": "causes",
                "description": "A causes B.",
                "source_concepts": ["a"],
                "target_concepts": ["b"],
            }
        ],
    }


def test_load_and_filter_relation_domain():
    ontology = load_ontology(valid_data())
    assert ontology.valid_relations("a", "b")[0].id == "causes"
    assert ontology.valid_relations("b", "a") == ()


def test_loads_grounding_terms():
    data = valid_data()
    data["concepts"][0]["grounding_terms"] = ["storm", "flood"]
    data["relations"][0]["grounding_terms"] = ["cause"]

    ontology = load_ontology(data)

    assert ontology.concept_by_id["a"].grounding_terms == ("storm", "flood")
    assert ontology.relation_by_id["causes"].grounding_terms == ("cause",)


@pytest.mark.parametrize("bad_version", [0, 2, "1", None])
def test_rejects_unsupported_version(bad_version):
    data = valid_data()
    data["version"] = bad_version
    with pytest.raises(OntologyError):
        load_ontology(data)


def test_rejects_unknown_domain_concept():
    data = valid_data()
    data["relations"][0]["source_concepts"] = ["missing"]
    with pytest.raises(OntologyError, match="unknown concepts"):
        load_ontology(data)


def test_reference_ontology_stays_general():
    ontology = load_ontology("ontology.yaml")
    concept_ids = {concept.id for concept in ontology.concepts}

    assert len(ontology.concepts) == 23
    assert {
        "entity",
        "physical_entity",
        "abstraction",
        "person",
        "group",
        "organization",
        "location",
        "artifact",
        "natural_object",
        "animal",
        "plant",
        "substance",
        "food",
        "event",
        "act",
        "process",
        "state",
        "attribute",
        "communication",
        "cognition",
        "time_period",
        "quantity",
        "relation",
    } == concept_ids
    assert {"bridge", "city", "drought", "flood", "river", "storm"}.isdisjoint(concept_ids)
    assert "causes" in {
        relation.id for relation in ontology.valid_relations("event", "state")
    }
