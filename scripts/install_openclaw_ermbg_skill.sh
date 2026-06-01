#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${repo_root}/integrations/openclaw/ermbg-matte"
dst="${OPENCLAW_SKILLS_DIR:-${HOME}/.openclaw/workspace/skills}/ermbg-matte"

if [[ ! -d "${src}" ]]; then
  echo "missing source skill: ${src}" >&2
  exit 1
fi

mkdir -p "$(dirname "${dst}")"
rm -rf "${dst}"
cp -R "${src}" "${dst}"
chmod +x "${dst}/scripts/ermbg_matte.py"

echo "Installed ERMBG OpenClaw skill:"
echo "${dst}"
echo
echo "Smoke command:"
echo "python3 ${dst}/scripts/ermbg_matte.py --image /path/to/input.png"
