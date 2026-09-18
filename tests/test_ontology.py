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


def test_reference_ontology_has_fine_grained_entities():
    ontology = load_ontology("ontology.yaml")
    concept_ids = {concept.id for concept in ontology.concepts}

    assert len(ontology.concepts) == 48
    assert {
        "bridge",
        "city",
        "construction",
        "drought",
        "evacuation",
        "flood",
        "laboratory",
        "river",
        "sensor",
        "shelter",
        "storm",
        "village",
    } <= concept_ids
    assert ontology.concept_by_id["river"].grounding_terms == ("river",)
    assert "causes" in {
        relation.id for relation in ontology.valid_relations("storm", "flood")
    }
