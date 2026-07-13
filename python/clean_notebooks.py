from __future__ import annotations

import json
from pathlib import Path


def clean_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8-sig")

    return changed


def main() -> None:
    notebook_paths = sorted(Path("notebook").rglob("*.ipynb"))
    changed_paths = [path for path in notebook_paths if clean_notebook(path)]

    if changed_paths:
        for path in changed_paths:
            print(f"cleaned {path}")
    else:
        print("No notebook state to clean.")


if __name__ == "__main__":
    main()
