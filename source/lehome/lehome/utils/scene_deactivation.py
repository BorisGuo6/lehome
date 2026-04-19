from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def deactivate_subtrees_below_root(
    stage,
    root_path: str,
    keep_prim_path_prefixes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Deactivate non-whitelisted subtrees below ``root_path``.

    A prim subtree stays active if:
    - the prim path matches one of ``keep_prim_path_prefixes``; or
    - one of its descendants matches the whitelist, in which case only the
      non-whitelisted sibling subtrees are deactivated.
    """

    keep_prefixes = tuple(
        _normalize_path(path)
        for path in (keep_prim_path_prefixes or ())
        if _normalize_path(path)
    )
    deactivated_paths: list[str] = []
    kept_paths: list[str] = []
    bridge_paths: list[str] = []
    already_inactive_paths: list[str] = []

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return {
            "root_found": False,
            "root_path": root_path,
            "processed_count": 0,
            "deactivated_count": 0,
            "kept_count": 0,
            "bridge_count": 0,
            "already_inactive_count": 0,
            "deactivated_paths": [],
            "kept_paths": [],
            "bridge_paths": [],
            "already_inactive_paths": [],
        }

    processed_count = 0

    def walk(prim) -> bool:
        nonlocal processed_count
        if not prim.IsValid():
            return False
        if not prim.IsActive():
            already_inactive_paths.append(prim.GetPath().pathString)
            return False

        processed_count += 1
        prim_path = prim.GetPath().pathString

        if _is_kept_subtree_root(prim_path, keep_prefixes):
            kept_paths.append(prim_path)
            return True

        keep_due_to_descendant = False
        for child in prim.GetChildren():
            if walk(child):
                keep_due_to_descendant = True

        if keep_due_to_descendant:
            bridge_paths.append(prim_path)
            return True

        prim.SetActive(False)
        deactivated_paths.append(prim_path)
        return False

    for child in root_prim.GetChildren():
        walk(child)

    return {
        "root_found": True,
        "root_path": root_path,
        "processed_count": processed_count,
        "deactivated_count": len(deactivated_paths),
        "kept_count": len(kept_paths),
        "bridge_count": len(bridge_paths),
        "already_inactive_count": len(already_inactive_paths),
        "deactivated_paths": deactivated_paths,
        "kept_paths": kept_paths,
        "bridge_paths": bridge_paths,
        "already_inactive_paths": already_inactive_paths,
    }


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path:
        return path
    return path.rstrip("/")


def _is_kept_subtree_root(prim_path: str, keep_prefixes: Sequence[str]) -> bool:
    return any(
        prim_path == prefix or prim_path.startswith(f"{prefix}/")
        for prefix in keep_prefixes
    )
