# Hướng dẫn gán nhãn thủ công M3

Tài liệu này hướng dẫn một member chạy từ bước chọn publication đến khi có
file nhãn cuối cùng. Mỗi member chỉ cần thay `MEMBER_ID`; nếu dữ liệu M1/M2
được tổ chức khác cấu trúc chuẩn, chỉ cần sửa các biến đường dẫn tương ứng,
không cần sửa code Python.

## 1. Chuẩn bị

Chạy các lệnh từ thư mục gốc repository:

```bash
cd /path/to/arXiv-Project

# Chỉ thay giá trị này.
MEMBER_ID=24127023

# Các đường dẫn này có thể giữ nguyên nếu dùng layout chuẩn.
M1_ROOT=/path/to/data/m1
M2_ROOT=/path/to/data/m2
M3_ROOT=/path/to/data/m3-${MEMBER_ID}
PAPER_IDS_DIR="$PWD/input/member-paper-ids"
```

Range của member được lấy tự động từ `input/member_ranges.csv`:

```bash
RANGE_ID=$(awk -F, -v id="$MEMBER_ID" \
  '$1 == id { print $3; exit }' input/member_ranges.csv)

if [ -z "$RANGE_ID" ]; then
  echo "Không tìm thấy MEMBER_ID trong input/member_ranges.csv: $MEMBER_ID" >&2
  exit 1
fi

PAPER_IDS="$PAPER_IDS_DIR/$MEMBER_ID.txt"
BIBTEX="$M2_ROOT/$RANGE_ID/bibtex.jsonl"
REFERENCES="$M1_ROOT/$RANGE_ID"

test -f "$PAPER_IDS" || { echo "Thiếu file paper IDs: $PAPER_IDS" >&2; exit 1; }
test -f "$BIBTEX" || { echo "Thiếu file BibTeX: $BIBTEX" >&2; exit 1; }
test -d "$REFERENCES" || { echo "Thiếu thư mục references: $REFERENCES" >&2; exit 1; }

mkdir -p "$M3_ROOT/input-validation" \
  "$M3_ROOT/selection" \
  "$M3_ROOT/labeling/$RANGE_ID"

echo "member=$MEMBER_ID range=$RANGE_ID"
echo "bibtex=$BIBTEX"
echo "references=$REFERENCES"
```

### Layout dữ liệu khác nhau

`selection` không yêu cầu M1 và M2 phải nằm cùng một thư mục. Nếu member có
layout khác, giữ nguyên `MEMBER_ID` và gán trực tiếp hai biến sau:

```bash
BIBTEX=/duong-dan/thuc-te/toi/bibtex.jsonl
REFERENCES=/duong-dan/thuc-te/toi/thu-muc-chua-references.json
```

`BIBTEX` phải là một file `bibtex.jsonl`. Với lệnh `selection`,
`REFERENCES` phải là một thư mục chứa (trực tiếp hoặc lồng bên dưới) các file
`references.json`; chương trình sẽ tìm các file này đệ quy. Nếu member chỉ có
một file `references.json`, hãy đặt `REFERENCES` là thư mục cha của file đó.

## 2. Kiểm tra đầu vào M3

Nên chạy validation trước khi chọn publication:

```bash
python -m src.matching.validation \
  --bibtex "$BIBTEX" \
  --references "$REFERENCES" \
  --paper-ids "$PAPER_IDS" \
  --output "$M3_ROOT/input-validation/$RANGE_ID.json"
```

Mở file report và chỉ tiếp tục khi có:

```json
"status": "passed",
"error_count": 0
```

Cảnh báo M2 được giữ lại trong report nhưng không phải lỗi đầu vào M3. Nếu có
error, dừng lại và sửa đường dẫn hoặc dữ liệu trước khi gán nhãn.

## 3. Chọn publication để gán nhãn

```bash
python -m src.matching.selection \
  --bibtex "$BIBTEX" \
  --references "$REFERENCES" \
  --paper-ids "$PAPER_IDS" \
  --member-id "$MEMBER_ID" \
  --range-id "$RANGE_ID" \
  --output-dir "$M3_ROOT/selection/$RANGE_ID"
```

Kết quả gồm:

```text
$M3_ROOT/selection/<range-id>/data_inventory.csv
$M3_ROOT/selection/<range-id>/manual_publication_assignment_<member-id>.csv
```

Assignment có 6 publication:

- 3 publication `train`;
- 1 publication `valid`;
- 1 publication `test`;
- 1 publication `spare`.

Không tự thay đổi `arxiv_id` trong assignment. Nếu phải thay publication,
hãy trao đổi với người phụ trách M3 để bảo toàn split và audit trail.

## 4. Tạo template nhãn

```bash
ASSIGNMENT="$M3_ROOT/selection/$RANGE_ID/manual_publication_assignment_${MEMBER_ID}.csv"
LABEL_ROOT="$M3_ROOT/labeling/$RANGE_ID"

python -m src.matching.labeling prepare \
  --assignment "$ASSIGNMENT" \
  --bibtex "$BIBTEX" \
  --references "$REFERENCES" \
  --output "$LABEL_ROOT/labels_template.csv"

python -m src.matching.labeling prepare-review \
  --input "$LABEL_ROOT/labels_template.csv" \
  --output-dir "$LABEL_ROOT/review"
```

`prepare-review` tạo file tổng hợp:

```text
$LABEL_ROOT/review/labels_review.csv
```

Các dòng không có candidate arXiv được điền trước là `no_match`. Không xóa
dòng BibTeX nào khỏi template.

## 5. Gán nhãn bằng giao diện

Nếu chưa cài Streamlit:

```bash
python -m pip install "streamlit>=1.40,<2"
```

Khởi chạy giao diện:

```bash
streamlit run src/matching/labeling_app.py -- \
  --input "$LABEL_ROOT/review/labels_review.csv" \
  --output "$LABEL_ROOT/review/labels_completed.csv"
```

Mỗi dòng tương ứng với một BibTeX entry. Chỉ dùng các trạng thái:

- `match`: chọn đúng một `groundtruth_arxiv_id`;
- `no_match`: không có reference arXiv phù hợp, để ground truth rỗng;
- `uncertain`: chỉ dùng tạm trong lúc review, không được giữ trong file final.

Không nhập candidate giả, không dùng `None`, `null` hoặc chuỗi rỗng làm ID
reference. Nếu title BibTeX thiếu, dùng author/year/DOI và thông tin reference
để quyết định; không tự đoán title.

## 6. Validate nhãn cuối

Sau khi hoàn tất review:

```bash
python -m src.matching.labeling validate \
  --input "$LABEL_ROOT/review/labels_completed.csv" \
  --template "$LABEL_ROOT/labels_template.csv" \
  --output "$LABEL_ROOT/labels_final.csv" \
  --final
```

Chỉ dùng `labels_final.csv` cho các bước matching, feature engineering và
đánh giá. Nếu lệnh báo lỗi, không sửa trực tiếp file final để bỏ qua lỗi; hãy
sửa file review rồi chạy validate lại.

## 7. Nguồn dữ liệu của bước `selection`

Các chỉ số BibTeX trong inventory được đọc trực tiếp từ file truyền qua
`--bibtex` trong lệnh trên. Với layout chuẩn, đó là:

```text
$M2_ROOT/<range-id>/bibtex.jsonl
```

`selection` **không** tự tìm bản M2 mới nhất theo timestamp và không đọc các
nguồn BibTeX khác có sẵn. Nếu truyền nhầm đường dẫn tới M2 cũ, inventory cũng
sẽ được tính từ M2 cũ.

Trong `bibtex.jsonl`, code dùng các trường:

- `arxiv_id` để lọc theo `$PAPER_IDS`;
- `title`, `authors`, `year` để tính tỷ lệ thiếu metadata;
- mỗi dòng để đếm `bibtex_count`.

Candidate/reference statistics được đọc riêng từ `REFERENCES` của M1:

- tổng số reference;
- số `ref_arxiv_id` duy nhất;
- số reference arXiv bị trùng.

`selection` không lấy các chỉ số này từ `statistics.json`,
`validation_report.json`, `bibtex_provenance.jsonl` hoặc kết quả M3 cũ. Vì vậy,
để chắc chắn dùng M2 mới nhất, hãy kiểm tra biến `BIBTEX` trước khi chạy:

```bash
realpath "$BIBTEX"
head -n 1 "$BIBTEX"
```

## 8. Output cần bàn giao

Member gửi lại tối thiểu các file sau:

```text
input-validation/<range-id>.json
selection/<range-id>/data_inventory.csv
selection/<range-id>/manual_publication_assignment_<member-id>.csv
labeling/<range-id>/labels_template.csv
labeling/<range-id>/labels_final.csv
```

Giữ nguyên `MEMBER_ID`, `RANGE_ID`, assignment và template để nhóm có thể truy
vết chính xác dữ liệu đã dùng cho M3.
