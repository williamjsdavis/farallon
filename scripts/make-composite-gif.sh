#!/usr/bin/env bash
# Stack two screen recordings into one speeded-up GIF: a band of the first on
# top, a band of the second below, both starting at the same moment and played
# at the same speed so their pace can be compared directly.
#
# Usage:
#   ./scripts/make-composite-gif.sh [options] top.mov bottom.mov output.gif
#
# Options (defaults in brackets):
#   --duration T       source seconds to show from each start [until the shorter ends]
#   --top-start T      where the top recording's run starts [1]
#   --bottom-start T   where the bottom recording's run starts [1]
#   --top-rows A-B     rows of the top recording to keep [0-621]
#   --bottom-rows A-B  rows of the bottom recording to keep [158-646]
#   --ref-height N     frame height those rows are measured in [729, i.e. a
#                      1200px-wide frame, the size of the example screenshots]
#   --speed N          playback speedup [10]
#   --width N          output width in px [1200]
#   --fps N            timing granularity before duplicate frames are dropped [15]
#   --dither MODE      bayer | none [bayer]
#   --force            overwrite an existing output file
#
# Requires ffmpeg (brew install ffmpeg). FFMPEG=/path/to/ffmpeg overrides.
set -euo pipefail

cg_duration=
cg_top_start=1
cg_bottom_start=1
cg_top_rows=0-621
cg_bottom_rows=158-646
cg_ref_height=729
cg_speed=10
cg_width=1200
cg_fps=15
cg_dither=bayer
cg_force=0
cg_args=()

while (( $# )); do
  case "$1" in
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    --duration) cg_duration="$2"; shift 2 ;;
    --top-start) cg_top_start="$2"; shift 2 ;;
    --bottom-start) cg_bottom_start="$2"; shift 2 ;;
    --top-rows) cg_top_rows="$2"; shift 2 ;;
    --bottom-rows) cg_bottom_rows="$2"; shift 2 ;;
    --ref-height) cg_ref_height="$2"; shift 2 ;;
    --speed) cg_speed="$2"; shift 2 ;;
    --width) cg_width="$2"; shift 2 ;;
    --fps) cg_fps="$2"; shift 2 ;;
    --dither) cg_dither="$2"; shift 2 ;;
    --force) cg_force=1; shift ;;
    --) shift; cg_args+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) cg_args+=("$1"); shift ;;
  esac
done

(( ${#cg_args[@]} == 3 )) || { echo "Usage: $0 [options] top.mov bottom.mov output.gif" >&2; exit 1; }
cg_top="${cg_args[0]}"
cg_bottom="${cg_args[1]}"
cg_output="${cg_args[2]}"
for f in "$cg_top" "$cg_bottom"; do
  [[ -f "$f" ]] || { echo "Input not found: $f" >&2; exit 1; }
done
if [[ -e "$cg_output" ]] && (( ! cg_force )); then
  echo "Output already exists: $cg_output (pass --force to overwrite)" >&2
  exit 1
fi

cg_ffmpeg="${FFMPEG:-ffmpeg}"
command -v "$cg_ffmpeg" >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install ffmpeg or set FFMPEG=/path/to/ffmpeg." >&2
  exit 1
}

# Accept 90, 1:30 or 00:01:30 wherever a timestamp is expected.
cg_seconds() {
  awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; printf "%.3f", s }' <<<"$1"
}

# Turn "A-B" rows of the reference frame into a resolution-independent crop,
# applied to the full-size source before scaling so it is resampled only once.
cg_crop() {
  local a="${1%-*}" b="${1#*-}"
  (( b > a )) || { echo "Bad row range: $1" >&2; exit 1; }
  printf 'crop=iw:ih*%d/%d:0:ih*%d/%d' "$(( b - a ))" "$cg_ref_height" "$a" "$cg_ref_height"
}

cg_band() { # crop -> "filter chain for one band"
  printf '%s,setpts=(PTS-STARTPTS)/%s,fps=%s,scale=%s:-1:flags=lanczos,setsar=1' \
    "$1" "$cg_speed" "$cg_fps" "$cg_width"
}

cg_top_args=(-ss "$(cg_seconds "$cg_top_start")")
cg_bottom_args=(-ss "$(cg_seconds "$cg_bottom_start")")
if [[ -n "$cg_duration" ]]; then
  cg_top_args+=(-t "$(cg_seconds "$cg_duration")")
  cg_bottom_args+=(-t "$(cg_seconds "$cg_duration")")
fi

case "$cg_dither" in
  bayer) cg_paletteuse='paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' ;;
  none) cg_paletteuse='paletteuse=dither=none:diff_mode=rectangle' ;;
  *) echo "Unknown --dither: $cg_dither (want bayer or none)" >&2; exit 1 ;;
esac

cg_work_dir="$(mktemp -d "${TMPDIR:-/tmp}/farallon-composite.XXXXXX")"
cg_frames="$cg_work_dir/frames.mkv"
cg_palette="$cg_work_dir/palette.png"
trap 'rm -rf -- "$cg_work_dir"' EXIT

# Pass 1: decode both sources once, stack the bands, and keep only frames where
# something changed in either band (see make-gif.sh for why this matters).
# shortest=1 ends the stack when the shorter band runs out, so neither side
# ever sits frozen while the other keeps moving.
"$cg_ffmpeg" -n -hide_banner -loglevel error \
  "${cg_top_args[@]}" -i "$cg_top" "${cg_bottom_args[@]}" -i "$cg_bottom" \
  -filter_complex "[0:v]$(cg_band "$(cg_crop "$cg_top_rows")")[top];[1:v]$(cg_band "$(cg_crop "$cg_bottom_rows")")[bot];[top][bot]vstack=inputs=2:shortest=1,mpdecimate=hi=64*12:lo=64*5:frac=0.05" \
  -fps_mode vfr -an -c:v ffv1 "$cg_frames"

# Pass 2: one palette for both bands, weighting pixels that change.
"$cg_ffmpeg" -n -hide_banner -loglevel error -i "$cg_frames" \
  -vf 'palettegen=stats_mode=diff' -frames:v 1 -update 1 "$cg_palette"

# Pass 3: map to the palette and encode only the changed rectangles.
"$cg_ffmpeg" -y -hide_banner -loglevel error -i "$cg_frames" -i "$cg_palette" \
  -filter_complex "[0:v][1:v]$cg_paletteuse" \
  -an -loop 0 -gifflags +transdiff -fps_mode passthrough \
  -final_delay "${GIF_FINAL_DELAY:--1}" "$cg_output"

printf 'Created: %s (%s)\n' "$cg_output" "$(wc -c <"$cg_output" | awk '{ printf "%.1f MB", $1 / 1e6 }')"
