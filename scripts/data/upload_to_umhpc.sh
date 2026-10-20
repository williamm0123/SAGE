#!/usr/bin/env bash
# Upload the DiffusionForensics subsets SAGE needs (~2.2 GB of the 141 GB download) to UMHPC,
# verify them, and extract them there.
#
#   bash scripts/data/upload_to_umhpc.sh
#
# What is uploaded (archives as-is, NOT extracted locally):
#   images/test/lsun_bedroom/{real,sdv1_new,sdv2}.tar.gz   1000 + 1000 + 1000 images
#   images/train/lsun_bedroom/real.tar.gz                  40k LSUN real (DIRE training data; overlap check)
#   checkpoints/lsun_adm.pth                               DIRE classifier -> weights/dire/
#
# Robustness:
#   * one SSH master connection (ControlMaster): you type the UMHPC password once
#   * rsync --partial --append-verify: an interrupted file resumes where it stopped
#   * every file retried up to $RETRIES times with back-off; re-running the script skips finished files
#   * sha256 checked locally and on UMHPC before extraction; extraction is idempotent
#
# Overrides: SRC, REMOTE, REMOTE_ROOT, RETRIES, INCLUDE_TRAIN_REAL=0, SAGE_SSH (custom ssh command)
set -euo pipefail

SRC=${SRC:-/media/william/8EBAE4F4BAE4DA2B/BaiduNetdiskDownload/DiffusionForensics}
REMOTE=${REMOTE:-qinglong@login01.dicc.um.edu.my}
REMOTE_ROOT=${REMOTE_ROOT:-SAGE}            # relative to your UMHPC home, or absolute
RETRIES=${RETRIES:-20}
INCLUDE_TRAIN_REAL=${INCLUDE_TRAIN_REAL:-1}

FILES=(
  images/test/lsun_bedroom/real.tar.gz
  images/test/lsun_bedroom/sdv1_new.tar.gz
  images/test/lsun_bedroom/sdv2.tar.gz
  checkpoints/lsun_adm.pth
)
[ "$INCLUDE_TRAIN_REAL" = 1 ] && FILES+=(images/train/lsun_bedroom/real.tar.gz)

RAW=$REMOTE_ROOT/data/raw/DiffusionForensics
STAGE=$RAW/_archives                        # archives + checksums are kept here
CTL=${XDG_RUNTIME_DIR:-/tmp}/sage-umhpc-%r@%h:%p
if [ -z "${SAGE_SSH:-}" ]; then
  SAGE_SSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=2h -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
fi
rssh() { $SAGE_SSH "$REMOTE" "$@"; }

cd "$SRC"
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing $SRC/$f" >&2; exit 1; }
done

echo "== [1/5] local sha256 =="
SUMS=$(mktemp)
trap 'rm -f "$SUMS"' EXIT
sha256sum "${FILES[@]}" | tee "$SUMS"
du -ch "${FILES[@]}" | tail -1

echo "== [2/5] connect to $REMOTE (enter password once) =="
rssh "mkdir -p '$STAGE' && echo connected: \$(hostname) && df -h . | tail -1"

echo "== [3/5] upload (resumable) =="
for f in "${FILES[@]}"; do
  dst_dir="$STAGE/$(dirname "$f")"
  rssh "mkdir -p '$dst_dir'"
  for ((i = 1; i <= RETRIES; i++)); do
    if rsync -a --partial --append-verify --info=progress2 -e "$SAGE_SSH" "$f" "$REMOTE:$dst_dir/"; then
      break
    fi
    if [ "$i" -eq "$RETRIES" ]; then
      echo "giving up on $f after $RETRIES attempts; re-run the script to resume" >&2
      exit 1
    fi
    wait_s=$(( i < 6 ? 10 * i : 60 ))
    echo "transfer of $f interrupted (attempt $i/$RETRIES); retrying in ${wait_s}s ..." >&2
    sleep "$wait_s"
  done
done

echo "== [4/5] verify sha256 on UMHPC =="
rsync -a -e "$SAGE_SSH" "$SUMS" "$REMOTE:$STAGE/SHA256SUMS"
rssh "cd '$STAGE' && sha256sum -c SHA256SUMS"

echo "== [5/5] extract on UMHPC =="
rssh bash -s -- "$RAW" "$REMOTE_ROOT" <<'REMOTE_SCRIPT'
set -euo pipefail
cd ~
raw=$(realpath -m "$1"); root=$(realpath -m "$2")   # relative paths are relative to $HOME
cd "$raw/_archives"
for a in $(find images -name '*.tar.gz' | sort); do
  dest="$raw/$(dirname "$a")"
  marker="$dest/.extracted_$(basename "$a")"
  if [ -f "$marker" ]; then echo "already extracted: $a"; continue; fi
  mkdir -p "$dest"
  tar -xf "$a" -C "$dest"      # tar detects gzip itself (some DIRE ".tar.gz" files are plain tar)
  top=$(tar -tf "$a" | head -1 | cut -d/ -f1)
  n=$(find "$dest/$top" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
  echo "$a -> $dest/$top ($n images)"
  touch "$marker"
done
mkdir -p "$root/weights/dire"
[ -f "$root/weights/dire/lsun_adm.pth" ] || cp checkpoints/lsun_adm.pth "$root/weights/dire/lsun_adm.pth"
sha256sum "$root/weights/dire/lsun_adm.pth"
echo
echo "Done. Next on UMHPC (see docs/DATA.md):"
echo "  python scripts/data/build_manifest.py --real $raw/images/test/lsun_bedroom/real \\"
echo "    --sdv1 $raw/images/test/lsun_bedroom/sdv1_new --sdv2 $raw/images/test/lsun_bedroom/sdv2 \\"
echo "    --exclude $raw/images/train/lsun_bedroom/real --dataset-version '...' --out \$DATA_ROOT/sage_v1"
REMOTE_SCRIPT

$SAGE_SSH -O exit "$REMOTE" 2>/dev/null || true
