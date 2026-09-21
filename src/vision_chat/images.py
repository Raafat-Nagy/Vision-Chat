import base64
import mimetypes
from pathlib import Path


def image_to_data_url(image_path: str) -> str:
    """Read a local image file and encode it as a base64 data URL."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")

    image_data = base64.b64encode(path.read_bytes()).decode("utf-8")

    return f"data:{mime_type};base64,{image_data}"
