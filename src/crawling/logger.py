import logging
from logging.handlers import RotatingFileHandler

from src.crawling.config import settings

# 1. CẤU HÌNH ĐỊNH DẠNG LOG
# =========================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

APP_LOGGER_NAME = "arxiv_project"

# 2. HÀM KHỞI TẠO LOGGER
# =========================================================

def setup_logging() -> None:
    """
    Khởi tạo hệ thống logging cho dự án.

    Log được ghi ra:
    - Console
    - File <settings.log_dir>/arxiv.log

    Cấu hình được đọc từ src/crawling/config.py.
    """

    # 2.1. Tạo thư mục log nếu chưa tồn tại.
    settings.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 2.2. Kiểm tra mức log.
    level_name = settings.log_level.upper()
    level = logging.getLevelName(level_name)

    if not isinstance(level, int):
        raise ValueError(
            f"Log level không hợp lệ: {settings.log_level}"
        )

    # 2.3. Lấy logger gốc riêng của ứng dụng.
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    app_logger.setLevel(level)

    # Không chuyển log của ứng dụng lên root logger.
    app_logger.propagate = False

    # 2.4. Xóa handler cũ do ứng dụng quản lý.
    # Giúp tránh ghi log lặp khi setup_logging() được gọi lại.
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
        handler.close()

    # 2.5. Tạo formatter dùng chung.
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # 2.6. Handler ghi ra console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 2.7. Handler ghi ra file, có xoay vòng dung lượng.
    file_handler = RotatingFileHandler(
        filename=settings.log_dir / "arxiv.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # 2.8. Đăng ký các handler.
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)

    app_logger.info("Logging system initialized")

# 3. HÀM LẤY LOGGER THEO MODULE
# =========================================================

def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger cho một module trong dự án.

    Ví dụ:
        logger = get_logger(__name__)
    """

    # Loại bỏ tiền tố package để tên log không phụ thuộc cấu trúc thư mục.
    for prefix in ("src.crawling.", "src."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    logger_name = (
        f"{APP_LOGGER_NAME}.{name}"
        if name
        else APP_LOGGER_NAME
    )

    return logging.getLogger(logger_name)
