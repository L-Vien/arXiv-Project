import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_path(env_name: str, default: Path) -> Path:
    """Đọc đường dẫn từ biến môi trường hoặc dùng giá trị mặc định."""
    value = os.getenv(env_name)
    path = Path(value).expanduser() if value else default

    # Chuẩn hóa thành đường dẫn tuyệt đối. Đường dẫn tương đối do người dùng cung cấp được tính từ thư mục gốc của repository.
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()

@dataclass(frozen=True)
class Settings:
    # Corpus nằm ngoài repository mã nguồn.
    data_dir: Path = get_path(
        "ARXIV_DATA_DIR",
        PROJECT_ROOT.parent / "data" / "corpus",
    )

    # Kết quả chạy thử và các lần benchmark.
    output_dir: Path = get_path(
        "ARXIV_OUTPUT_DIR",
        PROJECT_ROOT / "outputs" / "run-001",
    )

    # Logging
    log_dir: Path = get_path(
        "ARXIV_LOG_DIR",
        PROJECT_ROOT / "logs",
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # HTTP request
    request_timeout: float = 60.0
    max_retries: int = 3

    # API pacing: khoảng chờ giữa các request
    arxiv_api_delay: float = 3.0
    semantic_scholar_delay: float = 1.0

    # Nhận diện client khi gọi API
    user_agent: str = os.getenv(
        "ARXIV_USER_AGENT",
        "ArxivProject/1.0",
    )

    # URL dịch vụ
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    semantic_scholar_api_url: str = (
        "https://api.semanticscholar.org/graph/v1"
    )

settings = Settings()
