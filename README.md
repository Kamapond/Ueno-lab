# Mistral OCR sample code

from __future__ import annotations
import os, json, base64, hashlib
from pathlib import Path
from typing import Iterable, Any, List, Dict
from datetime import datetime


# ===== ライブラリ =====
import httpx
from mistralai import Mistral
from dotenv import load_dotenv, find_dotenv

# ===== 入出力フォルダ =====
INPUT_DIR = Path(
    r"XXXXXXXXXXXXXXXXXXXXXXX"
)
OUTPUT_DIR = INPUT_DIR.parent / "_ocr_json"
OUTPUT_JSON = OUTPUT_DIR / "combined_pages.json"
RECURSIVE = False
TEST_LIMIT: int | None = None

# ===== Mistral OCR =====
MODEL_NAME = "mistral-ocr-latest"
INCLUDE_IMAGE_BASE64 = False

# ===== 対象拡張子 =====
EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"]


# -------- Utils --------
def iter_images(dir_path: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        for ext in EXTS:
            yield from dir_path.rglob(f"*{ext}")
    else:
        for ext in EXTS:
            yield from dir_path.glob(f"*{ext}")

def file_sha256(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def to_data_uri(img_path: Path) -> str:
    b = img_path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    ext = img_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext in (".png",):
        mime = "image/png"
    elif ext in (".tif", ".tiff"):
        mime = "image/tiff"
    else:
        mime = "application/octet-stream"
    return f"data:{mime};base64,{b64}"

def extract_markdown_from_ocr_response(ocr_response: Any) -> str:
    """
    client.ocr.process(...) の返り値から pages[].markdown を結合して返す。
    """
    pages = []
    if isinstance(ocr_response, dict):
        pages = ocr_response.get("pages") or []
    else:
        pages = getattr(ocr_response, "pages", []) or []
    chunks = []
    for p in pages:
        if isinstance(p, dict):
            md = p.get("markdown", "") or ""
        else:
            md = getattr(p, "markdown", "") or ""
        if md:
            chunks.append(md)
    return "\n\n".join(chunks).strip()


# -------- Main --------
def main():
    # .env 読み込み（Agent\.env を最優先）
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY が見つかりません。Agent\\.env に設定してください。")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # SDK クライアント
    client = Mistral(api_key=api_key)

    # 画像列挙
    images = sorted(iter_images(INPUT_DIR, recursive=RECURSIVE), key=lambda p: p.name.lower())
    if TEST_LIMIT:
        images = images[:TEST_LIMIT]
    if not images:
        print(f"画像が見つかりませんでした: {INPUT_DIR}")
        return

    print(f"検出ファイル数: {len(images)}")

    # 重複排除（内容ハッシュで判定）
    unique_images: List[Path] = []
    seen_hashes: set[str] = set()
    for p in images:
        try:
            h = file_sha256(p)
        except Exception as e:
            print(f"[SKIP] ハッシュ計算失敗: {p} -> {e}")
            continue
        if h in seen_hashes:
            print(f"[DEDUP] 同一内容のためスキップ: {p.name}")
            continue
        seen_hashes.add(h)
        unique_images.append(p)

    print(f"OCR対象（重複除去後）: {len(unique_images)}")

    combined_pages: List[Dict[str, Any]] = []

    for idx, img in enumerate(unique_images, start=1):
        # ログは相対表記（任意）
        try:
            rel = img.relative_to(INPUT_DIR)
        except Exception:
            rel = img.name
        print(f"[OCR] ({idx}/{len(unique_images)}) {rel}")

        try:
            data_uri = to_data_uri(img)
            ocr_response = client.ocr.process(
                model=MODEL_NAME,
                document={"type": "image_url", "image_url": data_uri},
                include_image_base64=INCLUDE_IMAGE_BASE64,
            )
        except Exception as e:
            print(f"  -> SKIP: API ERROR: {e}")
            # 失敗時はそのページを飛ばす（必要なら空ページを挿入してもよい）
            continue

        md_text = extract_markdown_from_ocr_response(ocr_response)
        if not md_text:
            print("  -> WARN: markdown empty")

        combined_pages.append({
            "page": idx,
            "markdown": md_text
        })

    # 統合JSONを書き出し（最小構成）
    combined_payload = {
        "summary": { "count": len(combined_pages) },
        "pages": combined_pages
    }
    OUTPUT_JSON.write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完了: {len(combined_pages)}/{len(unique_images)} 件 -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
