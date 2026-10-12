"""Offline tests for the download helpers (no network)."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from collect_downloads import looks_like_a_multitrack
from fetch_cambridge import is_complete_zip, read_urls, target_name


class FetchTests(unittest.TestCase):
    def test_target_name(self):
        self.assertEqual(target_name("https://mtkarchive.cambridge-mt.com/Song_Full.zip"), "Song_Full.zip")
        self.assertEqual(target_name("https://x.test/a/b/My%20Song_Full.zip"), "My Song_Full.zip")
        with self.assertRaises(ValueError):
            target_name("https://x.test/page.html")

    def test_read_urls_skips_comments_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "urls.txt")
            path.write_text("# pick guitar-heavy songs\nhttps://x.test/a.zip\n\nhttps://x.test/a.zip\n"
                            "https://x.test/b.zip\n")
            self.assertEqual(read_urls(path), ["https://x.test/a.zip", "https://x.test/b.zip"])
            self.assertEqual(read_urls(Path(tmp, "missing.txt")), [])
            path.write_text("http://x.test/a.zip\n")
            with self.assertRaises(ValueError):
                read_urls(path)

    def test_zip_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, junk, silent = Path(tmp, "g.zip"), Path(tmp, "j.zip"), Path(tmp, "s.zip")
            with zipfile.ZipFile(good, "w") as archive:
                archive.writestr("Song/01_ElecGtr.wav", b"RIFF....WAVE")
            with zipfile.ZipFile(silent, "w") as archive:
                archive.writestr("notes.txt", b"hello")
            junk.write_bytes(b"not a zip at all" * 100)
            self.assertTrue(is_complete_zip(good))
            self.assertFalse(is_complete_zip(junk))
            self.assertFalse(is_complete_zip(Path(tmp, "missing.zip")))
            self.assertTrue(looks_like_a_multitrack(good))
            self.assertFalse(looks_like_a_multitrack(silent))
            self.assertFalse(looks_like_a_multitrack(junk))


if __name__ == "__main__":
    unittest.main()
