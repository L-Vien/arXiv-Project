## **DEADLINE THỨ 5, NGÀY 12/11/2026**

## **TV1 — Technical lead, integration, export và benchmark**

**Người phụ trách:** Lan Viên

### **Nhiệm vụ**

* Chốt và duy trì technical contract.  
* Viết pipeline điều phối.  
* Viết data export.  
* Viết validation.  
* Viết statistics.  
* Viết benchmark.  
* Tích hợp Jupyter Book.  
* Review interface của các thành viên.  
* Không viết toàn bộ nội dung report thay cho cả nhóm.

### **File đề xuất**

```
src/parsing/
├── contracts.py
├── pipeline.py
├── export.py
├── validation.py
├── statistics.py
└── benchmark.py
```

### **Input**

```
Kết quả của TV2, TV3, TV4, TV5
```

### **Output**

```
elements
hierarchy
bibtex
statistics.json
benchmark.json
validation_report.json
```

### **Lưu ý**

TV1 có thể làm ngay bằng mock:

```
mock_source_bundle()
mock_normalized_tree()
mock_reference_bundle()
```

Không cần chờ các module thật.

---

## **TV2 — Multi-file gathering và source loading**

**Người phụ trách:** Minh Thư

### **Nhiệm vụ**

* Tìm file chứa `\documentclass`.  
* Theo dõi `\input` và `\include`.  
* Xử lý include lồng nhau.  
* Bỏ qua include trong comment.  
* Resolve path tương đối.  
* Xử lý phần mở rộng `.tex`.  
* Phát hiện cycle.  
* Bỏ qua file dư thừa.  
* Ghi nhận file lỗi hoặc không tồn tại.

### **File đề xuất**

```
src/parsing/compilation_path.py
tests/parsing/test_compilation_path.py
```

### **Input**

```
Một thư mục version từ corpus Milestone 1
```

### **Output nên dùng**

Không nên chỉ trả về một chuỗi `tex`. Nên dùng:

```
SourceBundle(
    arxiv_id="2501.01234",
    version=1,
    status="ok",
    root_file="main.tex",
    files=[
        SourceFile(
            path="main.tex",
            order=0,
            content="..."
        ),
        SourceFile(
            path="sections/intro.tex",
            order=1,
            content="..."
        )
    ],
    diagnostics=[]
)
```

Có thể có thêm `expanded_text`, nhưng không được bỏ mất danh sách file gốc.

### **Lý do**

Thông tin file gốc cần thiết cho:

* debug;  
* xử lý path;  
* truy nguyên lỗi;  
* xử lý bibliography;  
* thống kê file;  
* kiểm tra compilation path.

---

## **TV3 — Hierarchical parser**

**Người phụ trách:** Bảo Nhi

### **Nhiệm vụ**

* Nhận diện document.  
* Nhận diện section, subsection, paragraph.  
* Xử lý section có đánh số và `\section*`.  
* Xử lý itemize, enumerate, item lồng nhau.  
* Nhận diện block formula, figure, table ở mức cấu trúc.  
* Giữ Acknowledgements và Appendix.  
* Loại phần References khỏi hierarchy.  
* Bảo toàn thứ tự node.

### **File đề xuất**

```
src/parsing/hierarchy.py
tests/parsing/test_hierarchy.py
```

### **Input**

```
SourceBundle
```

### **Output**

```
RawTree(
    arxiv_id="2501.01234",
    version=1,
    root=Node(
        kind="document",
        content=None,
        children=[...]
    )
)
```

### **Ranh giới với TV4**

TV3 chỉ cần nhận diện:

```
document
heading
paragraph
list
list_item
block_formula
figure
```

TV3 không cần hoàn thiện việc:

* tách paragraph thành sentence;  
* chuẩn hóa math;  
* xóa formatting command;  
* tạo element ID cuối cùng.

---

## **TV4 — Standardization và full-text deduplication**

**Người phụ trách:** Quốc Học

### **Nhiệm vụ**

* Xóa LaTeX formatting không có ý nghĩa.  
* Chuẩn hóa whitespace.  
* Chuẩn hóa inline math.  
* Chuẩn hóa block math.  
* Tách sentence.  
* Bảo vệ URL, decimal, abbreviation, citation và math.  
* Chuyển table/figure về schema cuối.  
* Chuẩn hóa citation key trong nội dung.  
* Deduplicate element giữa các version của cùng publication.  
* Sinh `element_id`.  
* Tạo mapping cho hierarchy.

### **File đề xuất**

```
src/parsing/normalization.py
src/parsing/deduplication.py
tests/parsing/test_normalization.py
tests/parsing/test_deduplication.py
```

### **Input**

```
RawTree
citation_map
```

### **Output**

```
NormalizedVersion
DeduplicatedPublication
```

### **Kind cuối cùng nên chốt**

```
document
heading
list
list_item
sentence
block_formula
figure
```

`paragraph` chỉ là node trung gian, không được giữ làm leaf cuối cùng.

### **Quy tắc dedup**

* Dedup sau cleanup.  
* Chỉ dedup trong cùng publication.  
* Hai publication khác nhau không dùng chung `element_id`.  
* Chỉ full-text match chính xác, không dùng similarity.  
* Nếu hai node có cùng content nhưng khác `kind`, không tự động gộp.  
* ID phải deterministic.

Ví dụ:

```
2501-01234#root-v1
2501-01234#h1
2501-01234#s1
```

---

## **TV5 — Reference extraction và BibTeX deduplication**

**Người phụ trách:** Bá Đạt

### **Nhiệm vụ**

* Trích xuất `\bibitem`.  
* Đọc `.bib` nếu có.  
* Có thể xử lý `.bbl` như fallback nếu cần.  
* Chuẩn hóa entry type, title, authors, year, DOI, journal, URL.  
* Phát hiện reference trùng giữa các version.  
* Gộp field.  
* Chọn citation key canonical.  
* Tạo citation map theo từng version.  
* Rewrite các citation key.

### **File đề xuất**

```
src/parsing/references.py
src/parsing/citation_rewrite.py
tests/parsing/test_references.py
```

### **Input**

```
SourceBundle của tất cả version trong một publication
```

### **Output**

```
ReferenceBundle(
    arxiv_id="2501.01234",
    entries=[...],
    citation_maps={
        1: {
            "transformer": "vaswani2017"
        },
        2: {
            "vaswani2017": "vaswani2017"
        }
    }
)
```

### **Quy tắc chọn key**

Cần chọn một quy tắc deterministic. Đề xuất:

1. Nếu cùng DOI hoặc arXiv ID thì gộp.  
2. Nếu không có thì so sánh title đã normalize \+ authors \+ year.  
3. Chọn key xuất hiện ở version sớm nhất.  
4. Nếu vẫn trùng, chọn key theo alphabet.  
5. Gộp tất cả field, không làm mất dữ liệu.

---

# **3\. Pipeline chạy thực tế**

Trong lúc phát triển, các module làm song song. Khi chạy thật, pipeline nên là:

```
TV2: gather all versions
        │
        ├── TV3: parse structure
        └── TV5: extract references
                    │
                    └── citation maps
                              ↓
                    TV4: normalize + deduplicate
                              ↓
                    TV1: export + validate + benchmark
```

Điểm quan trọng:

* TV3 và TV5 có thể chạy song song sau khi có `SourceBundle`.  
* TV4 chỉ cần citation map khi chuẩn hóa citation trong text.  
* TV1 có thể viết pipeline bằng mock trước.  
* Không ai phải chờ implementation thật của người khác để bắt đầu.

Có một bước integration cuối cùng là bắt buộc, nhưng đó là bước ghép hệ thống, không phải lý do để trì hoãn việc phát triển riêng.

---

# **4\. Technical contract cần chốt**

## **Bảng `elements`**

| Cột | Kiểu |
| ----- | ----- |
| `element_id` | string, non-null |
| `arxiv_id` | string, non-null |
| `kind` | dictionary/string |
| `content` | string |

## **Bảng `hierarchy`**

| Cột | Kiểu |
| ----- | ----- |
| `arxiv_id` | string |
| `version` | int |
| `element_id` | string |
| `parent_id` | string |

## **Bảng `bibtex`**

| Cột | Kiểu |
| ----- | ----- |
| `arxiv_id` | string |
| `citation_key` | string |
| `entry_type` | string |
| `title` | string |
| `authors` | list\[string\] |
| `year` | int/null |
| `doi` | string/null |
| `journal` | string/null |
| `url` | string/null |

## **Status chung**

```
ok
empty_no_tex
root_not_found
missing_include
partial_parse
parse_error
bibtex_partial
```

Không được âm thầm bỏ qua lỗi.

---

# **5\. Bộ test chung**

Nên giữ toàn bộ 10 test case trong kế hoạch bạn gửi:

| ID | Tình huống | Module chính |
| ----- | ----- | ----- |
| TC01 | Một file `.tex` | TV2 |
| TC02 | Include lồng nhau, comment, file dư | TV2 |
| TC03 | Section, subsection, appendix | TV3 |
| TC04 | Equation, figure, table, itemize | TV3 |
| TC05 | URL, decimal, abbreviation, math | TV4 |
| TC06 | Hai version có nội dung giống nhau | TV4 |
| TC07 | Citation key khác nhau | TV5 |
| TC08 | BibTeX trùng nhưng field khác nhau | TV5 |
| TC09 | Version không có TeX | TV1, TV2 |
| TC10 | Nhiều version, hierarchy khác nhau | TV1, TV4 |

Mỗi test nên có:

```
input fixture
expected output
expected status
expected statistics
```

---

# **6\. Quy định làm việc tối ưu**

## **Contract**

* Cả nhóm chốt `contracts.py` và tài liệu contract trước khi code.  
* Sau khi chốt, tăng version contract nếu có thay đổi lớn.  
* Không tự ý đổi tên hàm, kiểu trả về hoặc schema.  
* Mọi thay đổi interface phải được cả nhóm duyệt.

## **Git**

Nên dùng:

```
feature/m2-integration
feature/m2-compilation-path
feature/m2-hierarchy
feature/m2-normalization
feature/m2-references
```

Không nên để nhiều người cùng sửa trực tiếp một nhánh `integration`.

Mỗi Pull Request phải có:

* code;  
* unit test;  
* fixture;  
* mô tả interface;  
* mô tả giới hạn;  
* kết quả test.

## **Repository**

Theo cấu trúc hiện tại của nhóm, nên dùng:

```
src/parsing/
tests/parsing/
```

Không commit:

* corpus;  
* output lớn;  
* log;  
* cache;  
* virtual environment;  
* API key.

## **Report**

Mỗi thành viên tự viết phần của mình:

| Phần report | Người viết |
| ----- | ----- |
| Kiến trúc pipeline, benchmark tổng hợp | TV1 |
| Multi-file gathering | TV2 |
| Hierarchical parsing | TV3 |
| Normalization và full-text deduplication | TV4 |
| Reference extraction và deduplication | TV5 |

TV1 chỉ biên tập và thống nhất, không viết lại toàn bộ report.

---

# **7\. Bảng phân công cuối cùng**

| Thành viên | Phụ trách chính | Input | Output |
| ----- | ----- | ----- | ----- |
| TV1 — Lan Viên | Contract, integration, export, validation, benchmark | Kết quả các module | Ba bảng, metrics, validation report |
| TV2 — Minh Thư | Compilation path | Version directory | `SourceBundle` |
| TV3 — Bảo Nhi | Hierarchical parsing | `SourceBundle` | `RawTree` |
| TV4 — Quốc Học | Normalization, sentence splitting, full-text dedup | `RawTree`, citation map | Normalized tree, elements, hierarchy |
| TV5 — Bá Đạt | Reference extraction, BibTeX dedup, citation rewrite | `SourceBundle` | BibTeX entries, citation maps |

Đây là phương án cân bằng nhất vì:

* mỗi thành viên có một module kỹ thuật rõ ràng;  
* tất cả đều có thể phát triển bằng mock;  
* không có thành viên chỉ làm report;  
* TV4 giữ toàn bộ logic element identity, tránh phân tán dedup;  
* TV5 giữ toàn bộ logic citation, tránh xung đột key mapping;  
* TV1 kiểm soát schema, output và khả năng tái lập;  
* cấu trúc phù hợp với Milestone 2 và định hướng hiện tại của repository

`contracts.py` là “luật giao tiếp” chung giữa 5 module. File này không chứa thuật toán parse cụ thể; nó chỉ định nghĩa:

* Dữ liệu đầu vào và đầu ra.  
* Kiểu dữ liệu.  
* Tên trạng thái.  
* Các loại node.  
* Schema ba bảng cuối.  
* Quy tắc ID.  
* Interface public giữa các module.  
* Quy tắc lỗi và tính deterministic.

Với repository hiện tại, nên đặt tại:

```
src/parsing/contracts.py
```

---

# **1\. Những nội dung cả nhóm bắt buộc phải chốt**

## **1.1. Version của contract**

```
CONTRACT_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
```

Mọi thay đổi làm ảnh hưởng đến interface hoặc schema phải tăng version contract và thông báo cho cả nhóm.

---

## **1.2. Quy ước định danh**

Cần thống nhất:

```
arxiv_id: ID gốc, không có hậu tố version
version: số nguyên
element_id: duy nhất trong phạm vi một publication
citation_key: duy nhất trong phạm vi một publication
```

Đề xuất:

```
arxiv_id = "2501.01234"
element_id = "2501-01234#s1"
element_id = "2501-01234#root-v1"
```

Quy tắc:

* `element_id` phải deterministic.  
* Chạy lại cùng dữ liệu phải tạo cùng ID.  
* Element giống nhau giữa nhiều version dùng chung ID.  
* Hai publication khác nhau không dùng chung ID.  
* Root của mỗi version có ID riêng.

---

## **1.3. Danh sách trạng thái**

Các module phải dùng chung một bộ status:

```
ok
empty_no_tex
root_not_found
missing_include
partial_parse
parse_error
bibtex_partial
error
```

Đề xuất không dùng quá nhiều status tùy ý trong từng module.

---

## **1.4. Danh sách `kind`**

Cần phân biệt node trung gian và node cuối:

```
document
heading
paragraph
list
list_item
sentence
block_formula
figure
```

Quy định:

* `paragraph` chỉ là node trung gian.  
* Leaf cuối cùng chỉ gồm `sentence`, `block_formula`, `figure`.  
* Table được lưu dưới `kind="figure"`.  
* `list` và `list_item` là node phân nhánh.  
* `References` không xuất hiện trong cây.  
* `Acknowledgements` và `Appendix` vẫn xuất hiện.

---

# **2\. Các cấu trúc dữ liệu trung gian**

## **2.1. Diagnostic**

Dùng chung để các module ghi nhận lỗi hoặc cảnh báo.

```
@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    file_path: str | None = None
    line: int | None = None
```

Severity:

```
info
warning
error
```

Ví dụ:

```
Diagnostic(
    severity=DiagnosticSeverity.WARNING,
    code="MISSING_INCLUDE",
    message="Included file was not found",
    file_path="sections/method.tex",
)
```

---

## **2.2. SourceFile**

TV2 không nên chỉ trả về một chuỗi LaTeX lớn. Cần giữ lại thông tin file nguồn.

```
@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    content: str
    order: int
```

`order` là thứ tự file trong compilation path.

---

## **2.3. SourceBundle**

Đây là output của TV2 và input của TV3, TV5.

```
@dataclass(frozen=True)
class SourceBundle:
    arxiv_id: str
    version: int
    status: ProcessingStatus
    root_file: str | None
    files: tuple[SourceFile, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
```

Ý nghĩa:

| Trường | Ý nghĩa |
| ----- | ----- |
| `arxiv_id` | ID publication |
| `version` | Version hiện tại |
| `status` | Trạng thái gathering |
| `root_file` | File chứa `\documentclass` |
| `files` | Các file thực sự được include |
| `diagnostics` | Warning/error |

Nếu version không có TeX:

```
SourceBundle(
    arxiv_id="2501.01234",
    version=2,
    status=ProcessingStatus.EMPTY_NO_TEX,
    root_file=None,
    files=(),
)
```

---

## **2.4. TreeNode**

TV3 tạo cây thô, TV4 tiếp tục chuẩn hóa cây này.

```
@dataclass(frozen=True)
class TreeNode:
    kind: NodeKind
    content: str | None = None
    children: tuple["TreeNode", ...] = ()
    level: int | None = None
    source_file: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
```

Quy định:

* `children` luôn giữ đúng thứ tự xuất hiện.  
* `level` chủ yếu dùng cho heading.  
* `source_file` dùng để debug.  
* `content` của `RawTree` là LaTeX chưa chuẩn hóa.  
* `content` của `NormalizedTree` là nội dung canonical.

---

## **2.5. VersionTree**

```
@dataclass(frozen=True)
class VersionTree:
    arxiv_id: str
    version: int
    root: TreeNode
    status: ProcessingStatus
    diagnostics: tuple[Diagnostic, ...] = ()
```

Đây là output của TV3.

---

## **2.6. BibEntry**

```
@dataclass(frozen=True)
class BibEntry:
    citation_key: str
    entry_type: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str = ""
    journal: str = ""
    url: str = ""
```

Nên thống nhất:

* Field text bị thiếu dùng `""`.  
* `year` bị thiếu dùng `None`.  
* `authors` luôn là tuple/list string.  
* Không tự tạo dữ liệu bibliographic không có trong source.

---

## **2.7. ReferenceBundle**

Đây là output của TV5.

```
@dataclass(frozen=True)
class ReferenceBundle:
    arxiv_id: str
    entries: tuple[BibEntry, ...]
    citation_maps: Mapping[int, Mapping[str, str]]
    diagnostics: tuple[Diagnostic, ...] = ()
```

Ví dụ:

```
citation_maps = {
    1: {
        "transformer": "vaswani2017",
    },
    2: {
        "vaswani2017": "vaswani2017",
    },
}
```

Citation map phải phân biệt theo version.

---

# **3\. Các cấu trúc dữ liệu đầu ra**

## **3.1. ElementRow**

```
@dataclass(frozen=True)
class ElementRow:
    element_id: str
    arxiv_id: str
    kind: str
    content: str
```

Schema tương ứng:

```
elements(
    element_id,
    arxiv_id,
    kind,
    content
)
```

---

## **3.2. HierarchyRow**

```
@dataclass(frozen=True)
class HierarchyRow:
    arxiv_id: str
    version: int
    element_id: str
    parent_id: str
```

Quy định:

* `element_id` phải tồn tại trong `elements`.  
* `parent_id` phải tồn tại trong `elements`.  
* Root có thể tồn tại trong `elements` nhưng không có parent, nên không tạo hierarchy row cho root.  
* Không được tạo cycle.  
* Một element có thể xuất hiện ở nhiều version với parent khác nhau.

---

## **3.3. BibtexRow**

```
@dataclass(frozen=True)
class BibtexRow:
    arxiv_id: str
    citation_key: str
    entry_type: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str
    journal: str
    url: str
```

Khi xuất sang Parquet/JSON, `authors` được chuyển thành list.

---

## **3.4. VersionManifestRow**

Đây là bảng phụ trợ nội bộ, rất cần thiết vì ba bảng chính không biểu diễn được version rỗng.

```
@dataclass(frozen=True)
class VersionManifestRow:
    arxiv_id: str
    version: int
    status: str
    root_file: str | None
    error_message: str | None = None
```

Có thể lưu thành:

```
version_manifest.jsonl
```

Bảng này giúp thống kê:

* Tổng số version.  
* Version không có TeX.  
* Version parse lỗi.  
* Version parse thành công.

---

# **4\. Interface giữa các module**

Các hàm public nên được chốt như sau:

```
def gather_version(
    source_dir: Path,
    arxiv_id: str,
    version: int,
) -> SourceBundle:
    ...
```

```
def parse_structure(
    source: SourceBundle,
) -> VersionTree:
    ...
```

```
def process_references(
    sources: Sequence[SourceBundle],
) -> ReferenceBundle:
    ...
```

```
def normalize_tree(
    tree: VersionTree,
    citation_map: Mapping[str, str] | None = None,
) -> VersionTree:
    ...
```

```
def deduplicate_publication(
    versions: Sequence[VersionTree],
    arxiv_id: str,
) -> tuple[
    tuple[ElementRow, ...],
    tuple[HierarchyRow, ...],
]:
    ...
```

```
def export_dataset(
    elements: Sequence[ElementRow],
    hierarchy: Sequence[HierarchyRow],
    bibtex: Sequence[BibtexRow],
    output_dir: Path,
) -> None:
    ...
```

Tên hàm có thể thay đổi, nhưng input/output phải giữ nguyên ý nghĩa.

---

# **5\. Thứ tự xử lý cần chốt**

Khi phát triển, các module chạy song song. Khi pipeline chạy thật, thứ tự nên là:

```
gather_version
      │
      ├── parse_structure
      └── process_references
                    │
                    ↓
             citation_maps
                    │
                    ↓
             normalize_tree
                    │
                    ↓
          deduplicate_publication
                    │
                    ↓
             export_dataset
```

Điểm quan trọng:

* TV3 và TV5 có thể chạy song song sau TV2.  
* TV4 nhận `citation_map` từ TV5.  
* Trong unit test, TV4 dùng citation map giả lập.  
* Không được dedup trước khi normalization hoàn tất.

---

# **6\. Quy tắc normalization phải ghi trong contract**

Cả nhóm cần chốt cụ thể:

| Vấn đề | Quy tắc |
| ----- | ----- |
| Line ending | Chuẩn hóa về `\n` |
| Whitespace | Xóa khoảng trắng dư thừa |
| Formatting command | Xóa command không mang ý nghĩa |
| Inline math | Dùng `$...$` |
| Block math | Dùng `\begin{equation}...\end{equation}` |
| Sentence | Không tách URL, decimal, abbreviation, math |
| Figure/table | Đều có `kind="figure"` |
| References | Không vào hierarchy |
| Acknowledgements | Giữ lại |
| Appendix | Giữ lại |
| Dedup | So sánh sau cleanup |
| Citation key | Rewrite theo citation map |

---

# **7\. Quy tắc lỗi**

Cả nhóm cần thống nhất:

* Lỗi một file không được làm mất toàn bộ publication nếu vẫn có thể xử lý phần còn lại.  
* Không âm thầm bỏ qua version.  
* Không tạo dữ liệu giả để che lỗi.  
* Mọi lỗi phải có `status` và `Diagnostic`.  
* Lỗi parse nghiêm trọng phải được ghi vào `version_manifest`.  
* Module chỉ raise exception cho lỗi lập trình hoặc lỗi hệ thống nghiêm trọng.

---

# **8\. Những gì không nên đưa vào `contracts.py`**

Không nên đặt các phần sau trong file contract:

* Regex cụ thể để parse LaTeX.  
* Thuật toán sentence splitting.  
* Logic tìm root.  
* Logic dedup.  
* Logic đọc/ghi file.  
* Code gọi API.  
* Code benchmark.  
* Logic riêng của Jupyter Book.  
* Các biến global có thể bị module khác thay đổi.

`contracts.py` chỉ định nghĩa “dữ liệu và luật giao tiếp”, không thực hiện nghiệp vụ.

