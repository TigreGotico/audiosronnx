"""Every Hub repo id constant in the engines points at the AudioSRonnx org."""
import ast
from pathlib import Path

ENGINES = Path(__file__).resolve().parent.parent / "audiosronnx" / "engines"


def _repo_ids():
    ids = {}
    for path in sorted(ENGINES.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_HF_REPO":
                        ids[path.stem] = node.value.value
    return ids


def test_repo_ids_use_new_org():
    ids = _repo_ids()
    assert len(ids) == 19
    for engine, repo_id in ids.items():
        assert repo_id.startswith("AudioSRonnx/"), (engine, repo_id)
        assert "TigreGotico/audiosronnx-" not in repo_id, (engine, repo_id)
