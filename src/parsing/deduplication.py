"""Khử trùng element giữa các version của cùng một publication (TV4, Milestone 2).

deduplicate_publication(versions, arxiv_id) nhận các NormalizedVersion (đã chuẩn hóa, theo contract của
LV) và trả DeduplicatedPublication gồm bảng elements và hierarchy.

Quy tắc:
  - Chỉ dedup trong cùng publication, khớp CHÍNH XÁC (kind, content) sau khi đã chuẩn hóa.
  - Node có nội dung (heading, sentence, block_formula, figure): cùng (kind, content) thì dùng chung
    element_id giữa các version. Nếu một nội dung lặp lại k lần trong cùng một version thì lần xuất
    hiện thứ i khớp với lần thứ i ở version khác (mỗi lần xuất hiện vẫn có id riêng).
  - Node cấu trúc không có nội dung (document, list, list_item): id riêng cho từng version.
  - Cùng nội dung nhưng khác kind thì không gộp.
  - ID deterministic: duyệt version tăng dần, duyệt node theo `order` (thứ tự tài liệu).
  - Hierarchy: mỗi node không phải root có một dòng (version, element_id, parent_id), xuất theo
    (version, order) nên thứ tự anh em được giữ.

  Ví dụ:  2501-01234#root-v1   2501-01234#h1   2501-01234#s1   2501-01234#bf1   2501-01234#f1
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
    """2501.01234 -> 2501-01234 ; hep-th/9901001 -> hep-th_9901001 (giống paper_dir_name của M1)."""
    return arxiv_id.replace(".", "-").replace("/", "_")


def deduplicate_publication(
    versions: Sequence[NormalizedVersion],
    arxiv_id: str,
) -> DeduplicatedPublication:
    """Các NormalizedVersion của một publication -> DeduplicatedPublication.

    Version không có node (empty_no_tex, ...) vẫn nằm trong `versions` của kết quả nhưng không sinh
    element hay hierarchy. Ném ValueError nếu dữ liệu sai hợp đồng: khác arxiv_id, trùng version,
    không có đúng một root, parent không tồn tại, hoặc kind không hợp lệ.
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
            raise ValueError(f"NormalizedVersion thuộc {nv.arxiv_id!r}, không phải {arxiv_id!r}")
        if nv.version in seen_versions:
            raise ValueError(f"Trùng version {nv.version} của {arxiv_id}")
        seen_versions.add(nv.version)
        if not nv.nodes:
            continue

        roots = [n for n in nv.nodes if n.parent_local_id is None]
        if len(roots) != 1 or roots[0].local_id != nv.root_local_id:
            raise ValueError(f"{arxiv_id} v{nv.version}: cần đúng một root khớp root_local_id")

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
                raise ValueError(f"Node kind {kind!r} không hợp lệ trong cây đã chuẩn hóa")

            if node.local_id in id_of:
                raise ValueError(f"{arxiv_id} v{nv.version}: local_id trùng {node.local_id!r}")
            id_of[node.local_id] = eid
            if node.parent_local_id is not None:
                parent_eid = id_of.get(node.parent_local_id)
                if parent_eid is None:
                    raise ValueError(
                        f"{arxiv_id} v{nv.version}: parent {node.parent_local_id!r} "
                        f"của {node.local_id!r} chưa xuất hiện trước nó"
                    )
                hierarchy.append(HierarchyRow(arxiv_id, nv.version, eid, parent_eid))

    return DeduplicatedPublication(
        arxiv_id=arxiv_id,
        versions=tuple(ordered_versions),
        elements=tuple(elements.values()),
        hierarchy=tuple(hierarchy),
    )


class Deduplicator:
    """Cài đặt giao thức contracts.PublicationDeduplicator."""

    def deduplicate(
        self, versions: Sequence[NormalizedVersion], arxiv_id: str
    ) -> DeduplicatedPublication:
        return deduplicate_publication(versions, arxiv_id)
