#!/usr/bin/env python3
"""
download_compress.py — Download GGUF model parts + compress to 1-bit (.bit1)

Downloads a multi-part GGUF model from HuggingFace (or local source) and
optionally compresses expert weights to 1-bit (Q1_0) format using the
compress_bit1 C tool.

Usage:
  # Download + compress (full pipeline)
  python download_compress.py hf://username/model-name --output ./model.bit1

  # Download only (no compression)
  python download_compress.py hf://username/model-name --download-only

  # Compress existing GGUF split
  python download_compress.py --compress-only E:/models/model-prefix

  # Download with resume
  python download_compress.py hf://username/model-name --resume

Features:
  - Resume interrupted downloads (skip existing parts by size check)
  - Progress bars for each part
  - Automatic compression after all parts downloaded
  - Works with HuggingFace or direct HTTP URLs
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def download_part(url: str, dest_path: str, expected_size: int = 0,
                  resume: bool = False) -> bool:
    """
    Download a single file from URL to dest_path.
    Returns True on success, False on failure.
    """
    import urllib.request
    import urllib.error

    dest = Path(dest_path)
    mode = "wb"
    downloaded = 0

    # Resume: check existing file
    if resume and dest.exists():
        existing_size = dest.stat().st_size
        if expected_size > 0 and existing_size >= expected_size:
            print(f"  ✓ Already complete ({fmt_bytes(existing_size)})")
            return True
        if existing_size > 0:
            # Partial download — resume
            mode = "ab"
            downloaded = existing_size
            print(f"  ↻ Resuming from {fmt_bytes(downloaded)} / {fmt_bytes(expected_size)}")

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = expected_size or int(resp.headers.get("Content-Length", 0))
            if downloaded > 0:
                total = total + downloaded  # Content-Length is remaining bytes

            chunk_size = 1024 * 1024  # 1 MB
            last_print = time.time()

            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Print progress every 2 seconds
                    now = time.time()
                    if now - last_print >= 2.0:
                        pct = 100.0 * downloaded / total if total > 0 else 0
                        speed = downloaded / (now - time.time() + 0.001)
                        print(f"\r  [{pct:5.1f}%] {fmt_bytes(downloaded)} / "
                              f"{fmt_bytes(total)} @ {fmt_bytes(int(speed))}/s",
                              end="", flush=True)
                        last_print = now

        print(f"\r  [100.0%] {fmt_bytes(downloaded)} — done          ")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def parse_hf_url(url: str) -> tuple:
    """
    Parse a HuggingFace URL into (repo_id, file_pattern).
    Supports formats:
      hf://username/repo
      hf://username/repo/file-pattern
      https://huggingface.co/username/repo
    """
    # Strip scheme
    if url.startswith("hf://"):
        path = url[5:]
    elif url.startswith("https://huggingface.co/"):
        path = url[22:]
    else:
        path = url

    # Split into repo and optional file pattern
    parts = path.split("/", 2)
    if len(parts) >= 2:
        repo = f"{parts[0]}/{parts[1]}"
        pattern = parts[2] if len(parts) > 2 else None
    else:
        repo = path
        pattern = None

    return repo, pattern


def get_hf_file_list(repo: str, pattern: str = None) -> list:
    """
    Get list of files from HuggingFace repo using the API.
    Returns list of (filename, size, url) tuples.
    """
    import json
    import urllib.request

    api_url = f"https://huggingface.co/api/models/{repo}"
    try:
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"✗ Failed to fetch repo info: {e}")
        return []

    siblings = data.get("siblings", [])
    files = []
    for sib in siblings:
        fname = sib.get("rfilename", "")
        fsize = sib.get("size", 0)

        # Filter by pattern
        if pattern and pattern not in fname:
            continue

        # Only GGUF files
        if not fname.endswith(".gguf"):
            continue

        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        files.append((fname, fsize, url))

    files.sort(key=lambda x: x[0])  # Sort by filename
    return files


def run_compress_bit1(prefix: str, output: str, bit1_tool: str) -> bool:
    """Run the compress_bit1 C tool."""
    print(f"\n  Compressing {prefix} → {output}")

    cmd = [bit1_tool, prefix, "--output", output]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        for line in result.stdout.split("\n"):
            if line.strip():
                print(f"  {line.strip()}")
        if result.stderr:
            for line in result.stderr.split("\n"):
                if line.strip():
                    print(f"  ! {line.strip()}")
        if result.returncode != 0:
            print(f"  ✗ compress_bit1 failed with code {result.returncode}")
            return False
        return True
    except FileNotFoundError:
        print(f"  ✗ compress_bit1 tool not found at: {bit1_tool}")
        print(f"    Build it first: gcc -O3 -march=native -flto -I./src "
              f"tools/compress_bit1.c src/gguf/gguf.c src/quant/quant.c "
              f"-o build/compress_bit1.exe -lm")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ compress_bit1 timed out (>2 hours)")
        return False
    except Exception as e:
        print(f"  ✗ compress_bit1 error: {e}")
        return False


def verify_bit1_file(path: str) -> bool:
    """Verify a .bit1 file has a valid header."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"CT1B":
                print(f"  ✗ Invalid magic: {magic}")
                return False
            import struct
            version = struct.unpack("<I", f.read(4))[0]
            n_layers = struct.unpack("<I", f.read(4))[0]
            n_experts = struct.unpack("<I", f.read(4))[0]
            print(f"  ✓ .bit1 header valid: v{version}, {n_layers} layers, "
                  f"{n_experts} experts, {fmt_bytes(os.path.getsize(path))}")
            return True
    except Exception as e:
        print(f"  ✗ Cannot read .bit1 file: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF model parts + compress to 1-bit (.bit1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download + compress (full pipeline)
  python download_compress.py hf://bartowski/DeepSeek-V4-Flash-0731-MXFP4 \\
      --output ./model.bit1

  # Download only (no compression)
  python download_compress.py hf://bartowski/DeepSeek-V4-Flash-0731-MXFP4 \\
      --download-only

  # Compress existing GGUF split
  python download_compress.py --compress-only E:/models/model-prefix

  # Download with resume
  python download_compress.py hf://username/repo --resume
""")

    # Source
    parser.add_argument("source", nargs="?",
                        help="HuggingFace URL (hf://user/repo) or local prefix")

    # Modes
    parser.add_argument("--download-only", action="store_true",
                        help="Download only, skip compression")
    parser.add_argument("--compress-only", metavar="PREFIX",
                        help="Compress existing GGUF split (no download)")

    # Options
    parser.add_argument("--output", default=None,
                        help="Output .bit1 file path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted downloads")
    parser.add_argument("--bit1-tool", default=None,
                        help="Path to compress_bit1.exe (auto-detect if omitted)")
    parser.add_argument("--download-dir", default=None,
                        help="Directory to store downloaded GGUF parts")

    args = parser.parse_args()

    # ── Find compress_bit1 tool ──
    bit1_tool = args.bit1_tool
    if not bit1_tool:
        candidates = [
            "build/compress_bit1.exe",
            "../build/compress_bit1.exe",
            "compress_bit1.exe",
        ]
        script_dir = Path(__file__).parent
        for c in candidates:
            p = script_dir / c
            if p.exists():
                bit1_tool = str(p.resolve())
                break
            # Also check relative to cwd
            p2 = Path(c)
            if p2.exists():
                bit1_tool = str(p2.resolve())
                break

    # ── Compress-only mode ──
    if args.compress_only:
        prefix = args.compress_only
        output = args.output or f"{prefix}.bit1"

        if not bit1_tool:
            print("✗ compress_bit1 tool not found. Build it first.")
            sys.exit(1)

        success = run_compress_bit1(prefix, output, bit1_tool)
        if success:
            verify_bit1_file(output)
        sys.exit(0 if success else 1)

    # ── Download mode ──
    if not args.source:
        parser.print_help()
        sys.exit(1)

    source = args.source
    repo, pattern = parse_hf_url(source)

    print(f"📥 Fetching file list from: {repo}")
    files = get_hf_file_list(repo, pattern)

    if not files:
        print("✗ No GGUF files found in repo")
        sys.exit(1)

    print(f"   Found {len(files)} GGUF file(s):")
    total_size = 0
    for fname, fsize, _ in files:
        print(f"     • {fname}  ({fmt_bytes(fsize)})")
        total_size += fsize
    print(f"   Total: {fmt_bytes(total_size)}")

    # Download directory
    dl_dir = args.download_dir or f"./downloads/{repo.replace('/', '_')}"
    os.makedirs(dl_dir, exist_ok=True)

    # Determine prefix from downloaded files
    # The first part filename determines the prefix
    first_file = files[0][0]
    # e.g., "DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"
    # prefix = "DeepSeek-V4-Flash-0731-MXFP4"
    # Extract prefix by removing the part number
    import re
    prefix_match = re.match(r"^(.+?)-\d{5}-of-\d{5}\.gguf$", first_file)
    if prefix_match:
        prefix = prefix_match.group(1)
    else:
        prefix = first_file.replace(".gguf", "")

    # Download each part
    print(f"\n📥 Downloading to: {dl_dir}")
    all_ok = True
    for i, (fname, fsize, furl) in enumerate(files):
        dest = os.path.join(dl_dir, fname)
        print(f"\n  [{i+1}/{len(files)}] {fname}")
        ok = download_part(furl, dest, fsize, args.resume)
        if not ok:
            all_ok = False
            print(f"  ✗ Failed to download {fname}")
            continue

        # Verify size
        actual_size = os.path.getsize(dest) if os.path.exists(dest) else 0
        if fsize > 0 and actual_size < fsize:
            print(f"  ⚠ Size mismatch: got {fmt_bytes(actual_size)}, "
                  f"expected {fmt_bytes(fsize)}")
            all_ok = False

    if not all_ok:
        print("\n⚠ Some downloads had issues. Check above.")
        # Continue anyway — partial download may still be usable

    # ── Compress ──
    if args.download_only:
        print(f"\n✅ Download complete (--download-only, skipping compression)")
        print(f"   GGUF parts in: {dl_dir}")
        print(f"   To compress later: python {__file__} --compress-only "
              f"{os.path.join(dl_dir, prefix)}")
        sys.exit(0)

    if not bit1_tool:
        print("\n✗ compress_bit1 tool not found. Skipping compression.")
        print(f"  GGUF parts downloaded to: {dl_dir}")
        print(f"  Build the tool and run:")
        print(f"    python {__file__} --compress-only "
              f"{os.path.join(dl_dir, prefix)}")
        sys.exit(1)

    # Build the GGUF split prefix path
    gguf_prefix = os.path.join(dl_dir, prefix)
    output = args.output or f"{gguf_prefix}.bit1"

    print(f"\n🔧 Compressing to 1-bit...")
    success = run_compress_bit1(gguf_prefix, output, bit1_tool)
    if success:
        verify_bit1_file(output)
        print(f"\n✅ Pipeline complete!")
        print(f"   .bit1 file: {output}")
        print(f"   GGUF parts: {dl_dir}/")
    else:
        print(f"\n⚠ Compression failed. GGUF parts are in: {dl_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()