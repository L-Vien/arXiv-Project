import gzip
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.crawling.processor import (
    STATUS_ERROR, STATUS_MISSING, STATUS_NO_TEX, STATUS_OK,
    detect_source_type, process_version,
)


def make_tar(path: Path, files: dict[str, bytes], extra=None):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tar.addfile(info)


class TestProcessor(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.out = self.base / "out" / "v1"

    def tearDown(self):
        self._tmp.cleanup()

    def kept(self):
        return sorted(p.relative_to(self.out).as_posix()
                      for p in self.out.rglob("*") if p.is_file() and p.name != "source_status.json")

    def test_tar_filtering_keeps_tex_bib_bbl_only(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x", "refs.bib": b"y", "main.bbl": b"z", "my.sty": b"s",
                       "style.cls": b"c", "ref.bst": b"b",
                       "image.png": b"\x89PNG" * 100, "fig.pdf": b"%PDF", "script.py": b"p"})
        r = process_version(tar, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertEqual(self.kept(), ["main.bbl", "main.tex", "refs.bib"])
        self.assertFalse(tar.exists())                      # xử lý xong thì xóa archive

    def test_subdirs_and_uppercase_extension(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"sections/intro.TEX": b"x", "bib/refs.BIB": b"y", "figs/a.eps": b"e"})
        r = process_version(tar, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertEqual(self.kept(), ["bib/refs.BIB", "sections/intro.TEX"])

    def test_sizes_are_uncompressed_before_and_kept_after(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"a" * 1000, "big.png": b"b" * 5000})
        r = process_version(tar, self.out)
        self.assertEqual(r.raw_size, 6000)
        self.assertEqual(r.kept_size, 1000)
        self.assertEqual((r.before_size, r.after_size), (6000, 1000))
        self.assertLessEqual(r.after_size, r.before_size)

    def test_single_gz_tex(self):
        p = self.base / "single.gz"
        p.write_bytes(gzip.compress(b"\\documentclass{article}"))
        self.assertEqual(detect_source_type(p), "gz_single")
        r = process_version(p, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertEqual(self.kept(), ["main.tex"])

    def test_gz_wrapping_pdf_is_no_tex(self):
        p = self.base / "pdf.gz"
        p.write_bytes(gzip.compress(b"%PDF-1.5 binary"))
        r = process_version(p, self.out)
        self.assertEqual(r.status, STATUS_NO_TEX)
        self.assertEqual(self.kept(), [])

    def test_pdf_only_keeps_record(self):
        p = self.base / "x.pdf"
        p.write_bytes(b"%PDF-1.5 ...")
        r = process_version(p, self.out)
        self.assertEqual((r.status, r.source_type), (STATUS_NO_TEX, "pdf"))
        status = json.loads((self.out / "source_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], STATUS_NO_TEX)

    def test_docx_like_zip_is_no_tex(self):
        p = self.base / "doc.docx"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", "<xml/>")
        r = process_version(p, self.out)
        self.assertEqual((r.status, r.source_type), (STATUS_NO_TEX, "zip"))

    def test_only_bib_is_no_tex(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"refs.bib": b"y"})
        self.assertEqual(process_version(tar, self.out).status, STATUS_NO_TEX)

    def test_corrupt_archive_is_error_and_kept(self):
        p = self.base / "bad.tar.gz"
        p.write_bytes(b"\x1f\x8b" + b"not really gzip data")
        r = process_version(p, self.out)
        self.assertEqual(r.status, STATUS_ERROR)
        self.assertTrue(r.error)
        self.assertTrue(p.exists())                          # giữ lại để chạy lại
        self.assertTrue((self.out / "source_status.json").exists())

    def test_missing_file(self):
        r = process_version(self.base / "nope.tar.gz", self.out)
        self.assertEqual(r.status, STATUS_MISSING)
        self.assertTrue((self.out / "source_status.json").exists())

    def test_path_traversal_member_ignored(self):
        tar = self.base / "evil.tar.gz"
        make_tar(tar, {"main.tex": b"x", "../escape.tex": b"evil", "/abs.tex": b"evil"})
        r = process_version(tar, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertFalse((self.base / "out" / "escape.tex").exists())
        self.assertEqual(self.kept(), ["main.tex"])

    def test_symlink_member_ignored(self):
        info = tarfile.TarInfo("link.tex")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar = self.base / "l.tar.gz"
        make_tar(tar, {"main.tex": b"x"}, extra=[info])
        process_version(tar, self.out)
        self.assertEqual(self.kept(), ["main.tex"])

    def test_delete_archive_flag(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        process_version(tar, self.out, delete_archive=False)
        self.assertTrue(tar.exists())

    def test_skip_if_done_returns_previous_result(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        first = process_version(tar, self.out)
        self.assertEqual(first.status, STATUS_OK)
        (self.out / "main.tex").write_text("y", encoding="utf-8")   # cùng kích thước, nếu xử lý lại sẽ bị ghi đè thành "x"
        second = process_version(self.base / "a.tar.gz", self.out)  # archive đã bị xóa
        self.assertEqual(second.status, STATUS_OK)
        self.assertEqual(second.kept_files, ["main.tex"])
        self.assertEqual((self.out / "main.tex").read_text(encoding="utf-8"), "y")

    def test_skip_if_done_no_tex_also_skipped(self):
        p = self.base / "x.pdf"
        p.write_bytes(b"%PDF-1.5")
        process_version(p, self.out)
        self.assertEqual(process_version(p, self.out).status, STATUS_NO_TEX)

    def test_error_is_retried(self):
        bad = self.base / "a.tar.gz"
        bad.write_bytes(b"\x1f\x8b" + b"broken")
        self.assertEqual(process_version(bad, self.out).status, STATUS_ERROR)
        make_tar(bad, {"main.tex": b"x"})                      # lần sau tải lại được file đúng
        self.assertEqual(process_version(bad, self.out).status, STATUS_OK)

    def test_skip_if_done_false_reprocesses(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        process_version(tar, self.out)
        r = process_version(self.base / "a.tar.gz", self.out, skip_if_done=False)
        self.assertEqual(r.status, STATUS_MISSING)             # archive đã xóa -> xử lý thật sự chạy lại

    def test_skip_if_done_reprocesses_when_files_missing(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        process_version(tar, self.out, delete_archive=False)
        (self.out / "main.tex").unlink()                  # mất file sau khi đã ok
        r = process_version(tar, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertTrue((self.out / "main.tex").exists())

    def test_stale_tmp_dir_is_cleaned(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        stale = self.out.with_name(".v1.tmp")
        stale.mkdir(parents=True)
        (stale / "junk.tex").write_text("rác", encoding="utf-8")
        process_version(tar, self.out)
        self.assertFalse(stale.exists())
        self.assertEqual(self.kept(), ["main.tex"])

    def test_skip_if_done_reprocesses_when_size_changed(self):
        tar = self.base / "a.tar.gz"
        make_tar(tar, {"main.tex": b"x"})
        process_version(tar, self.out, delete_archive=False)
        (self.out / "main.tex").write_text("xxxx", encoding="utf-8")   # file bị hỏng/sửa
        r = process_version(tar, self.out)
        self.assertEqual(r.status, STATUS_OK)
        self.assertEqual((self.out / "main.tex").read_bytes(), b"x")   # đã giải nén lại

    def test_duplicate_members_counted_once(self):
        tar = self.base / "dup.tar.gz"
        with tarfile.open(tar, "w:gz") as t:
            for data in (b"old", b"newer"):               # hai member cùng tên main.tex
                info = tarfile.TarInfo("main.tex")
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        r = process_version(tar, self.out)
        self.assertEqual(r.kept_files, ["main.tex"])
        self.assertEqual(r.kept_size, 5)                  # bản sau ghi đè bản trước
        self.assertEqual((self.out / "main.tex").read_bytes(), b"newer")
        # version đã hoàn tất nên skip_if_done phải bỏ qua, dù archive đã bị xóa
        self.assertEqual(process_version(self.base / "gone.tar.gz", self.out).status, STATUS_OK)    

    
if __name__ == "__main__":
    unittest.main()
