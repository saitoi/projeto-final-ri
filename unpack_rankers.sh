#!/usr/bin/env bash
set -euo pipefail

command -v zstd >/dev/null || { echo "zstd not found"; exit 1; }
command -v tar  >/dev/null || { echo "tar not found";  exit 1; }

DEST="./rankers"
mkdir -p "$DEST"

if [ "$#" -gt 0 ]; then
  archives=( "$@" )
else
  shopt -s nullglob
  archives=( *.tar.zst )
fi

[ "${#archives[@]}" -gt 0 ] || { echo "no .tar.zst files"; exit 0; }

for f in "${archives[@]}"; do
  zstd -d --stdout -- "$f" | tar -xvf - -C "$DEST"
done
