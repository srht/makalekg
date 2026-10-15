from kg.discover import classify_funding
from kg.fetch import figure_key, parse_jats
from kg.infer.agent import WRITE_PATTERN
from kg.schema import ExtractionResult, claim_id, experiment_id, strict_json_schema
from kg.validate.consistency import parse_n_values


def test_strict_schema_is_closed():
    schema = strict_json_schema(ExtractionResult)
    assert "$defs" not in str(schema) and "$ref" not in str(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)


def test_ids_are_normalized():
    assert claim_id("Piezo1  is required.") == claim_id("piezo1 is required")
    assert experiment_id("10.1/X", "Figure 3", "d-h") == experiment_id("10.1/x", "Fig. 3", "d-h")


def test_figure_keys():
    assert figure_key("Fig. 3d") == "fig3" == figure_key("Figure 3")
    assert figure_key("Extended Data Fig. 2") == "edfig2"


def test_n_parsing():
    assert parse_n_values("n = 14, 14, 14 and 17 mice") == {14, 17}
    assert parse_n_values("n = 9–10 per group") == {9, 10}


def test_funding_classification():
    assert classify_funding(["National Institutes of Health"]) == "academic"
    assert classify_funding(["Infraredx, Inc."]) == "industry"
    assert classify_funding(["NIH", "Amgen"]) == "mixed"
    assert classify_funding([]) is None


def test_agent_cypher_guard():
    assert WRITE_PATTERN.search("MATCH (n) DETACH DELETE n")
    assert WRITE_PATTERN.search("merge (a:Gene {uid:'x'})")
    assert not WRITE_PATTERN.search("MATCH (a)-[r:SUPPORTS]->(c) RETURN c.text")


def test_jats_parsing():
    xml = """<article><front><article-meta><abstract><p>Abstract text.</p></abstract></article-meta></front>
    <body><sec><title>Results</title><p>Loading increased Hsp70 (Fig. 1a).</p>
    <fig id="f1"><label>Fig. 1</label><caption><p>a, Hsp70, n = 8 mice.</p></caption></fig></sec>
    <sec><title>Methods</title><p>Mice were loaded at 2 Hz.</p></sec></body></article>"""
    doc = parse_jats(xml, "10.1/x")
    kinds = [(s.name, s.kind) for s in doc.sections]
    assert ("Results", "body") in kinds and ("Methods", "methods") in kinds
    assert doc.legend_for("Fig. 1").text.startswith("Fig. 1 |")
    assert "n = 8" not in next(s.text for s in doc.sections if s.name == "Results")
