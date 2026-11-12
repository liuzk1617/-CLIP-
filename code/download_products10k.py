"""从 HF 镜像下载 Products-10K（nyris/products10k-v1），无需 Kaggle 账号。

镜像说明：hf-mirror.com 上的 nyris/products10k-v1 是 Products-10K 公开图像树的
parquet 打包版，含 train/test 两个 split、197,307 图、9,691 个 SKU。
字段: image(bytes) / img_file(相对路径) / sku(SKU编号字符串)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data" / "raw" / "products10k_v1"


def main() -> int:
    print(f"downloading to {LOCAL}", flush=True)
    path = snapshot_download(
        repo_id="nyris/products10k-v1",
        repo_type="dataset",
        local_dir=str(LOCAL),
        allow_patterns=["data/*.parquet", "README.md"],
        max_workers=4,
    )
    print("done:", path, flush=True)
    total = 0
    files = sorted((Path(path) / "data").glob("*.parquet"))
    for f in files:
        total += f.stat().st_size
    print(f"{len(files)} parquet files, {total/1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
