#!/usr/bin/env python3
import re
import shutil
import subprocess
import sys
import tempfile

PROPAGATING_OPS = ("Load", "AccessChain", "InBoundsAccessChain",
                   "PtrAccessChain", "CopyObject", "SampledImage")


def patch(spv_path):
    spirv_dis = shutil.which("spirv-dis")
    spirv_as = shutil.which("spirv-as")
    if not spirv_dis or not spirv_as:
        sys.stderr.write(f"[nonuniform_patch] spirv-dis/spirv-as not found; "
                         f"skipping {spv_path}\n")
        return 0

    text = subprocess.run([spirv_dis, "--raw-id", spv_path],
                          capture_output=True, text=True, check=True).stdout
    lines = text.splitlines()

    original = set(re.findall(r"OpDecorate\s+(%\w+)\s+NonUniform", text))
    nonuniform = set(original)

    changed = True
    while changed:
        changed = False
        for line in lines:
            m = re.match(r"\s*(%\w+)\s*=\s*Op(\w+)\b(.*)$", line)
            if not m:
                continue
            result, op, rest = m.group(1), m.group(2), m.group(3)
            if op not in PROPAGATING_OPS or result in nonuniform:
                continue
            if any(o in nonuniform for o in re.findall(r"%\w+", rest)):
                nonuniform.add(result)
                changed = True

    to_add = [i for i in nonuniform if i not in original]
    if not to_add:
        return 0

    last_decorate = max(i for i, l in enumerate(lines)
                        if re.match(r"\s*OpDecorate\b", l))
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if i == last_decorate:
            for i_id in to_add:
                out.append(f"               OpDecorate {i_id} NonUniform")

    with tempfile.NamedTemporaryFile("w", suffix=".spvasm", delete=False) as f:
        f.write("\n".join(out) + "\n")
        asm_path = f.name

    result = subprocess.run([spirv_as, "--target-env", "spv1.6",
                             asm_path, "-o", spv_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"[nonuniform_patch] spirv-as failed for {spv_path}:\n"
                         f"{result.stderr}\n")
        return 1
    sys.stderr.write(f"[nonuniform_patch] {spv_path}: decorated "
                     f"{len(to_add)} ids NonUniform\n")
    return 0


if __name__ == "__main__":
    sys.exit(patch(sys.argv[1]))
