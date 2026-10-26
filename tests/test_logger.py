import logging
import unittest

from src.logger import get_logger, setup_logging
from src.config import settings


class TestLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Chỉ khởi tạo logging một lần cho nhóm test.
        setup_logging()

    def test_get_logger_returns_logger(self):
        logger = get_logger("test_module")

        self.assertIsInstance(
            logger,
            logging.Logger,
        )

    def test_logger_names_use_project_namespace(self):
        logger = get_logger("src.downloader")

        self.assertEqual(
            logger.name,
            "arxiv_project.downloader",
        )

    def test_log_directory_exists(self):
        self.assertTrue(settings.log_dir.is_dir())

    def test_logger_uses_application_namespace(self):
        logger = get_logger("test_module")

        self.assertTrue(
            logger.name.startswith("arxiv_project.")
        )


if __name__ == "__main__":
    unittest.main()
