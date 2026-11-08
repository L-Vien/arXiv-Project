import unittest
from pathlib import Path

from src.crawling.config import PROJECT_ROOT, settings

class TestConfig(unittest.TestCase):

    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.is_dir())

    def test_data_dir_is_path(self):
        self.assertIsInstance(settings.data_dir, Path)

    def test_output_dir_is_path(self):
        self.assertIsInstance(settings.output_dir, Path)

    def test_log_dir_is_path(self):
        self.assertIsInstance(settings.log_dir, Path)

    def test_timeout_is_positive(self):
        self.assertGreater(settings.request_timeout, 0)

    def test_max_retries_is_non_negative(self):
        self.assertGreaterEqual(settings.max_retries, 0)

    def test_arxiv_delay_is_non_negative(self):
        self.assertGreaterEqual(settings.arxiv_api_delay, 0)

    def test_source_download_delay_defaults_to_reactive_only(self):
        self.assertEqual(settings.arxiv_api_delay, 0.0)

    def test_arxiv_rate_limit_cooldown_is_non_negative(self):
        self.assertGreaterEqual(settings.arxiv_rate_limit_cooldown, 0)

    def test_log_level_is_valid(self):
        valid_levels = {
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        }
        self.assertIn(settings.log_level, valid_levels)


if __name__ == "__main__":
    unittest.main()
