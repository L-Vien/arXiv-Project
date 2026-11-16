2. TV5 cần chỉnh sửa
File: [references.py](/Users/viennguyen/Documents/HCMUS/Nam3/HK1/Intro2DS/arXiv-Project/src/parsing/references.py)
A. Phân loại file đúng
Hiện tại .bbl đang được đưa vào _parse_bib_content, nhưng .bbl thực tế thường chứa \bibitem.
Cần đổi thành:
if path.endswith(".bib"):
    entries.extend(_parse_bib_content(content))
elif path.endswith(".bbl"):
    entries.extend(_parse_bibitem_content(content))
elif path.endswith(".tex"):
    entries.extend(_parse_bibitem_content(content))
B. Cải thiện .bib parser
Cần đảm bảo trích xuất:
citation_key
entry_type
title
authors
year
doi
journal
url
Đặc biệt:
parse author = {A and B};
không tách sai tên tổ chức có chữ and;
xử lý nested braces;
xử lý quoted value;
bỏ % comment;
hỗ trợ url, howpublished, note, eprint, \url{...}, \href{...}.
C. Cải thiện .bbl và \bibitem
.bbl và \bibitem cần tạo được entry hợp lệ.
Nếu không suy ra được một field thì dùng:
authors=()
year=None
doi=""
journal=""
url=""
Nhưng không được để toàn bộ author rỗng nếu source thật sự có thông tin tác giả.
Nếu parse lỗi, ghi diagnostic:
Diagnostic(
    severity=DiagnosticSeverity.WARNING,
    code="BIBTEX_PARTIAL",
    message="Could not fully parse bibliography entry",
    file_path="references.bbl",
)
D. Sửa quy tắc dedup reference
Quy tắc dedup hiện tại đang dùng title bị cắt 50 ký tự và chưa xét authors đầy đủ. Cần dùng:
1. DOI hoặc arXiv ID
2. normalized title + authors + year
3. nếu không đủ thông tin thì không tự động gộp
Không nên dùng:
_normalize_text(title)[:50]
vì hai title khác nhau có thể trùng 50 ký tự đầu.
E. Merge field
Khi hai entry trùng reference:
không làm mất title;
không làm mất authors;
không làm mất year;
không làm mất DOI;
không làm mất journal;
không làm mất URL.
Canonical key:
- ưu tiên key xuất hiện ở version sớm nhất;
- nếu cùng version thì chọn alphabet;
- deterministic.
Citation map phải tồn tại cho mọi version, kể cả version không có reference:
citation_maps = {
    1: {...},
    2: {},
    3: {...},
}
