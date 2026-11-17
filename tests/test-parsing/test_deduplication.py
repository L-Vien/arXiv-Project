import unittest

from src.parsing.contracts import (
    DeduplicatedPublication, NormalizedNode, NormalizedVersion, ProcessingStatus,
)
from src.parsing.deduplication import Deduplicator, deduplicate_publication, publication_key

AID = "2501.01234"


def n(kind, content="", *children):
    return (kind, content, children)


def nv(version, *children, arxiv_id=AID, status=ProcessingStatus.OK):
    """Dựng NormalizedVersion từ cây lồng nhau; node theo thứ tự duyệt trước."""
    nodes = []

    def add(spec, parent):
        kind, content, kids = spec
        order = len(nodes)
        local_id = f"n{order}"
        nodes.append(NormalizedNode(local_id, kind, content, parent, order))
        for kid in kids:
            add(kid, local_id)

    add(("document", "", children), None)
    return NormalizedVersion(arxiv_id, version, status, "n0", tuple(nodes), len(nodes))


def empty_version(version, status=ProcessingStatus.EMPTY_NO_TEX):
    return NormalizedVersion(AID, version, status, None, (), 0)


def ids_by_version(hierarchy):
    out = {}
    for h in hierarchy:
        out.setdefault(h.version, []).append(h.element_id)
    return out


class TestDedup(unittest.TestCase):
    def test_returns_deduplicated_publication(self):
        v1, v2 = nv(1, n("sentence", "A.")), nv(2, n("sentence", "A."))
        pub = deduplicate_publication([v2, v1], AID)
        self.assertIsInstance(pub, DeduplicatedPublication)
        self.assertEqual(pub.arxiv_id, AID)
        self.assertEqual(pub.versions, (v1, v2))      # sắp xếp theo version
        self.assertEqual(Deduplicator().deduplicate([v1, v2], AID), deduplicate_publication([v1, v2], AID))

    def test_tc06_identical_versions_share_element_ids(self):
        body = (n("heading", "Intro", n("sentence", "A."), n("sentence", "B.")),)
        pub = deduplicate_publication([nv(1, *body), nv(2, *body)], AID)
        by_v = ids_by_version(pub.hierarchy)
        self.assertEqual(by_v[1], by_v[2])
        self.assertEqual(by_v[1], ["2501-01234#h1", "2501-01234#s1", "2501-01234#s2"])
        ids = [e.element_id for e in pub.elements]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(i for i in ids if "root" in i), ["2501-01234#root-v1", "2501-01234#root-v2"])

    def test_hierarchy_rows_and_parents(self):
        pub = deduplicate_publication([nv(1, n("heading", "Intro", n("sentence", "A.")))], AID)
        self.assertEqual([(h.element_id, h.parent_id) for h in pub.hierarchy], [
            ("2501-01234#h1", "2501-01234#root-v1"),
            ("2501-01234#s1", "2501-01234#h1"),
        ])
        self.assertTrue(all(h.arxiv_id == AID and h.version == 1 for h in pub.hierarchy))

    def test_tc10_changed_hierarchy_between_versions(self):
        v1 = nv(1, n("heading", "Intro", n("sentence", "Shared.")), n("heading", "Method"))
        v2 = nv(2, n("heading", "Intro"), n("heading", "Method", n("sentence", "Shared."), n("sentence", "New.")))
        pub = deduplicate_publication([v1, v2], AID)
        parent = {(h.version, h.element_id): h.parent_id for h in pub.hierarchy}
        self.assertEqual(parent[(1, "2501-01234#s1")], "2501-01234#h1")   # Shared dưới Intro ở v1
        self.assertEqual(parent[(2, "2501-01234#s1")], "2501-01234#h2")   # Shared dưới Method ở v2
        self.assertEqual(len([e for e in pub.elements if e.kind == "sentence"]), 2)  # Shared + New

    def test_tc12_same_content_different_kind_not_merged(self):
        pub = deduplicate_publication(
            [nv(1, n("heading", "Results"), n("sentence", "Results"), n("figure", "Results"))], AID)
        self.assertEqual(sorted(e.kind for e in pub.elements if e.kind != "document"),
                         ["figure", "heading", "sentence"])
        self.assertEqual(len({h.element_id for h in pub.hierarchy}), 3)

    def test_repeated_sentence_in_one_version_gets_distinct_ids_but_matches_across_versions(self):
        v1 = nv(1, n("sentence", "Thanks."), n("sentence", "Other."), n("sentence", "Thanks."))
        v2 = nv(2, n("sentence", "Thanks."))
        pub = deduplicate_publication([v1, v2], AID)
        by_v = ids_by_version(pub.hierarchy)
        self.assertEqual(by_v[1], ["2501-01234#s1", "2501-01234#s2", "2501-01234#s3"])
        self.assertEqual(by_v[2], ["2501-01234#s1"])
        self.assertEqual([e.content for e in pub.elements if e.kind == "sentence"], ["Thanks.", "Other.", "Thanks."])

    def test_structural_nodes_are_per_version(self):
        lst = n("list", "", n("list_item", "", n("sentence", "One.")))
        pub = deduplicate_publication([nv(1, lst), nv(2, lst)], AID)
        by_v = ids_by_version(pub.hierarchy)
        self.assertEqual(by_v[1], ["2501-01234#l1-v1", "2501-01234#li1-v1", "2501-01234#s1"])
        self.assertEqual(by_v[2], ["2501-01234#l1-v2", "2501-01234#li1-v2", "2501-01234#s1"])

    def test_deterministic_and_independent_of_input_order(self):
        v1 = nv(1, n("sentence", "A."), n("sentence", "B."))
        v2 = nv(2, n("sentence", "B."), n("sentence", "C."))
        self.assertEqual(deduplicate_publication([v1, v2], AID), deduplicate_publication([v2, v1], AID))
        self.assertEqual(deduplicate_publication([v1, v2], AID), deduplicate_publication([v1, v2], AID))

    def test_tc09_version_without_tex_is_kept_in_versions_but_has_no_rows(self):
        pub = deduplicate_publication([nv(1, n("sentence", "A.")), empty_version(2)], AID)
        self.assertEqual({h.version for h in pub.hierarchy}, {1})
        self.assertFalse(any("root-v2" in e.element_id for e in pub.elements))
        self.assertEqual([v.version for v in pub.versions], [1, 2])
        self.assertEqual(pub.versions[1].status, ProcessingStatus.EMPTY_NO_TEX)

    def test_all_versions_empty_gives_no_rows(self):
        pub = deduplicate_publication([empty_version(1)], AID)
        self.assertEqual((pub.elements, pub.hierarchy), ((), ()))

    def test_old_style_arxiv_id(self):
        aid = "hep-th/9901001"
        pub = deduplicate_publication([nv(1, n("sentence", "A."), arxiv_id=aid)], aid)
        self.assertIn("hep-th_9901001#s1", [e.element_id for e in pub.elements])
        self.assertEqual(publication_key(aid), "hep-th_9901001")

    def test_two_publications_never_share_ids(self):
        a = deduplicate_publication([nv(1, n("sentence", "Same."))], AID)
        b = deduplicate_publication([nv(1, n("sentence", "Same."), arxiv_id="2502.00001")], "2502.00001")
        self.assertFalse({e.element_id for e in a.elements} & {e.element_id for e in b.elements})

    def test_contract_violations_raise(self):
        with self.assertRaises(ValueError):           # khác arxiv_id
            deduplicate_publication([nv(1, arxiv_id="2502.00001")], AID)
        with self.assertRaises(ValueError):           # trùng version
            deduplicate_publication([nv(1), nv(1)], AID)
        orphan = NormalizedVersion(AID, 1, ProcessingStatus.OK, "n0", (
            NormalizedNode("n0", "document", "", None, 0),
            NormalizedNode("n1", "sentence", "X.", "n9", 1),
        ), 2)
        with self.assertRaises(ValueError):           # parent không tồn tại
            deduplicate_publication([orphan], AID)
        two_roots = NormalizedVersion(AID, 1, ProcessingStatus.OK, "n0", (
            NormalizedNode("n0", "document", "", None, 0),
            NormalizedNode("n1", "document", "", None, 1),
        ), 2)
        with self.assertRaises(ValueError):           # hai root
            deduplicate_publication([two_roots], AID)

    def test_hierarchy_is_valid_tree(self):
        v = nv(1, n("heading", "A", n("heading", "B", n("sentence", "x."))), n("sentence", "y."))
        pub = deduplicate_publication([v], AID)
        known = {e.element_id for e in pub.elements}
        for h in pub.hierarchy:
            self.assertIn(h.element_id, known)
            self.assertIn(h.parent_id, known)
        children = [h.element_id for h in pub.hierarchy]
        self.assertEqual(len(children), len(set(children)))


if __name__ == "__main__":
    unittest.main()
