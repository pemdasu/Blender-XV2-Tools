from __future__ import annotations

import argparse
import ast
import re
import shutil
from pathlib import Path


def _read_version(src_init: Path) -> str:
    text = src_init.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Name)
                and target.id == "bl_info"
                and isinstance(node.value, ast.Dict)
            ):
                continue
            for key, val in zip(node.value.keys, node.value.values, strict=False):
                if not (
                    isinstance(key, ast.Constant)
                    and key.value == "version"
                    and isinstance(val, (ast.Tuple, ast.List))
                ):
                    continue
                parts = []
                for elt in val.elts:
                    if isinstance(elt, ast.Constant):
                        parts.append(str(elt.value))
                if parts:
                    return ".".join(parts)
    return "0.0.0"


def _write_changelog(version: str, root: Path, out_dir: Path) -> None:
    src_changelog = root / "CHANGELOG.md"
    if not src_changelog.exists():
        return

    content = src_changelog.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(rf"^##\s*{re.escape(version)}\s*$")
    start = None
    end = None
    for idx, line in enumerate(content):
        if pattern.match(line):
            start = idx
            break
    if start is not None:
        for idx in range(start + 1, len(content)):
            if content[idx].startswith("## "):
                end = idx
                break
        trimmed = content[start:end] if end else content[start:]
        text = "\n".join(trimmed).strip() + "\n"
    else:
        # Fallback: copy full file
        text = src_changelog.read_text(encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CHANGELOG.md").write_text(text, encoding="utf-8")


def _gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    patterns: list[str] = []
    if not gitignore.exists():
        return patterns
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def build(
    output_dir: Path | None = None,
    package_name: str = "blender_xv2_tools",
    version_override: str | None = None,
) -> Path:
    root = Path(__file__).parent
    src_dir = root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {src_dir}")

    version = version_override or _read_version(src_dir / "__init__.py")

    out_dir = output_dir or (root / "build")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_changelog(version, root, out_dir)

    target_dir = out_dir / package_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    patterns = _gitignore_patterns(root) + [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "Thumbs.db",
    ]
    patterns = list(dict.fromkeys(patterns))
    ignore = shutil.ignore_patterns(*patterns) if patterns else None
    shutil.copytree(src_dir, target_dir, ignore=ignore)

    zip_base = out_dir / f"{package_name}-{version}"
    zip_path = shutil.make_archive(
        base_name=str(zip_base),
        format="zip",
        root_dir=out_dir,
        base_dir=package_name,
    )

    if target_dir.exists():
        shutil.rmtree(target_dir)

    return Path(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Blender addon zip.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to place build artifacts (default: ./build)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="blender_xv2_tools",
        help="Addon folder/zip name (default: blender_xv2_tools)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string to embed in the zip name (default: read from bl_info).",
    )
    args = parser.parse_args()
    zip_path = build(args.output_dir, args.name, args.version)
    print(f"Finished: {zip_path}")


if __name__ == "__main__":
    main()
