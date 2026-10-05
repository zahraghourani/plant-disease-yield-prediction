"""
patch_ms_deform_attn.py

Re-applies the "_C is not defined" fix to GroundingDINO's
ms_deform_attn.py. Safe to run repeatedly (does nothing if already patched).

Why this is needed: on Windows the compiled CUDA extension (_C) is not
available. The original code only warns when the import fails, then still
takes the CUDA branch and crashes with  NameError: name '_C' is not defined.
The patch records whether the import worked (_C_AVAILABLE) and only uses
the CUDA kernel when it did; otherwise the pure-PyTorch fallback runs.

Run from the project root:
    python patch_ms_deform_attn.py
or pass an explicit path:
    python patch_ms_deform_attn.py path\\to\\ms_deform_attn.py
"""
import ast
import sys
from pathlib import Path

DEFAULT_PATH = Path("GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py")


def patch(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    raw = path.read_bytes().decode("utf-8")
    uses_crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")

    if "_C_AVAILABLE" in text:
        print("Already patched - nothing to do.")
        return 0

    out = []
    n_import = n_warn = n_if = 0
    for line in text.split("\n"):
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if "if torch.cuda.is_available() and value.is_cuda:" in line:
            line = line.replace(
                "if torch.cuda.is_available()",
                "if _C_AVAILABLE and torch.cuda.is_available()",
            )
            n_if += 1

        out.append(line)

        if stripped == "from groundingdino import _C":
            out.append(indent + "_C_AVAILABLE = True")
            n_import += 1
        elif stripped.startswith("warnings.warn(") and "custom C++ ops" in stripped:
            out.append(indent + "_C_AVAILABLE = False")
            n_warn += 1

    print(f"Edits found: import={n_import}, warn={n_warn}, cuda-branch={n_if}")
    if n_import != 1 or n_warn != 1 or n_if < 1:
        print("ERROR: file does not have the expected structure; nothing was written.")
        print("Open the file and apply the two edits manually (see chat instructions).")
        return 1

    patched = "\n".join(out)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched file would have a syntax error ({e}); nothing was written.")
        return 1

    if uses_crlf:
        patched = patched.replace("\n", "\r\n")
    path.write_bytes(patched.encode("utf-8"))
    print(f"Patched successfully: {path}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    sys.exit(patch(target))