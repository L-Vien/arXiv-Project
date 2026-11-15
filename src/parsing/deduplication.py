"""Publication-level deduplication of elements across versions (TV4, Milestone 2).

deduplicate_publication(versions, arxiv_id) takes the NormalizedVersions (already normalized, per LV's
contract) and returns a DeduplicatedPublication with the elements and hierarchy tables.

Rules:
  - Dedup only within one publication, by EXACT (kind, content) match after normalization.
  - Nodes with content (heading, sentence, block_formula, figure): the same (kind, content) shares one
    element_id across versions. If a content repeats k times inside one version, its i-th occurrence
    matches the i-th occurrence in another version (every occurrence still gets its own id).
  - Structural nodes without content (document, list, list_item): one id per version.
  - Same content but different kind is never merged.
  - IDs are deterministic: versions in ascending order, nodes by `order` (document order).
  - Hierarchy: one row (version, element_id, parent_id) per non-root node, emitted by (version, order)
    so sibling order is preserved.

  Examples: 2501-01234#root-v1   2501-01234#h1   2501-01234#s1   2501-01234#bf1   2501-01234#f1
            2501-01234#l1-v1 (list)   2501-01234#li1-v1 (list_item)
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.parsing.contracts import (
    DeduplicatedPublication, ElementRow, HierarchyRow, NodeKind, NormalizedVersion,
)

_CONTENT_PREFIX = {
    NodeKind.HEADING.value: "h",
    NodeKind.SENTENCE.value: "s",
    NodeKind.BLOCK_FORMULA.value: "bf",
    NodeKind.FIGURE.value: "f",
}
_STRUCT_PREFIX = {
    NodeKind.DOCUMENT.value: "d",
    NodeKind.LIST.value: "l",
    NodeKind.LIST_ITEM.value: "li",
}


def publication_key(arxiv_id: str) -> str:
    """2501.01234 -> 2501-01234 ; hep-th/9901001 -> hep-th_9901001 (same as paper_dir_name in M1)."""
    return arxiv_id.replace(".", "-").replace("/", "_")


def deduplicate_publication(
    versions: Sequence[NormalizedVersion],
    arxiv_id: str,
) -> DeduplicatedPublication:
    """NormalizedVersions of one publication -> DeduplicatedPublication.

    A version without nodes (empty_no_tex, ...) stays in the result's `versions` but produces no
    element or hierarchy rows. Raises ValueError on contract violations: different arxiv_id, duplicate
    version, not exactly one root, missing parent, or invalid kind.
    """
    pub = publication_key(arxiv_id)
    registry: dict[tuple[str, str, int], str] = {}
    counters: Counter[str] = Counter()
    elements: dict[str, ElementRow] = {}
    hierarchy: list[HierarchyRow] = []
    seen_versions: set[int] = set()
    ordered_versions = sorted(versions, key=lambda v: v.version)

    for nv in ordered_versions:
        if nv.arxiv_id != arxiv_id:
            raise ValueError(f"NormalizedVersion belongs to {nv.arxiv_id!r}, not {arxiv_id!r}")
        if nv.version in seen_versions:
            raise ValueError(f"Duplicate version {nv.version} of {arxiv_id}")
        seen_versions.add(nv.version)
        if not nv.nodes:
            continue

        roots = [n for n in nv.nodes if n.parent_local_id is None]
        if len(roots) != 1 or roots[0].local_id != nv.root_local_id:
            raise ValueError(f"{arxiv_id} v{nv.version}: expected exactly one root matching root_local_id")

        occurrences: Counter[tuple[str, str]] = Counter()
        struct_counters: Counter[str] = Counter()
        id_of: dict[str, str] = {}

        for node in sorted(nv.nodes, key=lambda n: n.order):
            kind = node.kind
            if node.parent_local_id is None:
                eid = f"{pub}#root-v{nv.version}"
                elements[eid] = ElementRow(eid, arxiv_id, NodeKind.DOCUMENT.value, "")
            elif kind in _CONTENT_PREFIX:
                occ_key = (kind, node.content)
                index = occurrences[occ_key]
                occurrences[occ_key] += 1
                key = (kind, node.content, index)
                eid = registry.get(key)
                if eid is None:
                    prefix = _CONTENT_PREFIX[kind]
                    counters[prefix] += 1
                    eid = f"{pub}#{prefix}{counters[prefix]}"
                    registry[key] = eid
                    elements[eid] = ElementRow(eid, arxiv_id, kind, node.content)
            elif kind in _STRUCT_PREFIX:
                prefix = _STRUCT_PREFIX[kind]
                struct_counters[prefix] += 1
                eid = f"{pub}#{prefix}{struct_counters[prefix]}-v{nv.version}"
                elements[eid] = ElementRow(eid, arxiv_id, kind, "")
            else:
                raise ValueError(f"Node kind {kind!r} is not valid in a normalized tree")

            if node.local_id in id_of:
                raise ValueError(f"{arxiv_id} v{nv.version}: duplicate local_id {node.local_id!r}")
            id_of[node.local_id] = eid
            if node.parent_local_id is not None:
                parent_eid = id_of.get(node.parent_local_id)
                if parent_eid is None:
                    raise ValueError(
                        f"{arxiv_id} v{nv.version}: parent {node.parent_local_id!r} "
                        f"of {node.local_id!r} does not appear before it"
                    )
                hierarchy.append(HierarchyRow(arxiv_id, nv.version, eid, parent_eid))

    return DeduplicatedPublication(
        arxiv_id=arxiv_id,
        versions=tuple(ordered_versions),
        elements=tuple(elements.values()),
        hierarchy=tuple(hierarchy),
    )


class Deduplicator:
    """Implements the contracts.PublicationDeduplicator protocol."""

    def deduplicate(
        self, versions: Sequence[NormalizedVersion], arxiv_id: str
    ) -> DeduplicatedPublication:
        return deduplicate_publication(versions, arxiv_id)
