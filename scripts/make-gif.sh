#!/usr/bin/env bash
# Turn a screen recording into a speeded-up GIF, spending frames only where the
# picture actually changes.
#
# Usage:
#   ./scripts/make-gif.sh [options] input.mov [output.gif]
#   ./scripts/make-gif.sh --analyze input.mov      # report static runs, suggest a trim
#
# Options (defaults in brackets):
#   --speed N        playback speedup [10]
#   --start T        trim: skip the first T (seconds or HH:MM:SS) [0]
#   --end T          trim: stop at T of the *source* timeline [end of file]
#   --width N        output width in px, height follows the aspect ratio [1200]
#   --fps N          timing granularity before duplicate frames are dropped [15]
#   --dither MODE    bayer | none [bayer]
#   --keep-dupes     disable duplicate-frame dropping (the old behaviour)
#   --force          overwrite an existing output file
#
# Requires ffmpeg (brew install ffmpeg). FFMPEG=/path/to/ffmpeg overrides.
set -euo pipefail

gif_speed=10
gif_start=0
gif_end=
gif_width=1200
gif_fps=15
gif_dither=bayer
gif_decimate=1
gif_force=0
gif_analyze=0
gif_args=()

while (( $# )); do
  case "$1" in
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    --analyze) gif_analyze=1; shift ;;
    --speed) gif_speed="$2"; shift 2 ;;
    --start) gif_start="$2"; shift 2 ;;
    --end) gif_end="$2"; shift 2 ;;
    --width) gif_width="$2"; shift 2 ;;
    --fps) gif_fps="$2"; shift 2 ;;
    --dither) gif_dither="$2"; shift 2 ;;
    --keep-dupes) gif_decimate=0; shift ;;
    --force) gif_force=1; shift ;;
    --) shift; gif_args+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) gif_args+=("$1"); shift ;;
  esac
done

(( ${#gif_args[@]} >= 1 )) || { echo "Usage: $0 [options] input.mov [output.gif]" >&2; exit 1; }
(( ${#gif_args[@]} <= 2 )) || { echo "Too many positional arguments" >&2; exit 1; }

gif_input="${gif_args[0]}"
[[ -f "$gif_input" ]] || { echo "Input not found: $gif_input" >&2; exit 1; }

gif_ffmpeg="${FFMPEG:-ffmpeg}"
# Reuse the encoder from this laptop's original conversion when still present.
gif_cached_encoder=/private/tmp/farallon-media-tools/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1
if [[ -z "${FFMPEG:-}" ]] && ! command -v ffmpeg >/dev/null 2>&1 && [[ -x "$gif_cached_encoder" ]]; then
  gif_ffmpeg="$gif_cached_encoder"
fi
command -v "$gif_ffmpeg" >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install ffmpeg or set FFMPEG=/path/to/ffmpeg." >&2
  exit 1
}

# Accept 90, 1:30 or 00:01:30 wherever a timestamp is expected.
gif_seconds() {
  awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; printf "%.3f", s }' <<<"$1"
}

# --analyze: list the runs where the picture holds still, so a trim can be
# chosen without opening the file in a player.
if (( gif_analyze )); then
  gif_total="$("$gif_ffmpeg" -hide_banner -i "$gif_input" 2>&1 |
    sed -n 's/.*Duration: \([0-9:.]*\),.*/\1/p' | head -1)"
  gif_total_s="$(gif_seconds "${gif_total:-0}")"
  printf 'Duration: %s (%.1fs)\n\nStatic runs of 8s or more:\n' "$gif_total" "$gif_total_s"
  "$gif_ffmpeg" -hide_banner -nostats -i "$gif_input" \
    -vf "scale=480:-1,freezedetect=n=0.003:d=8" -map 0:v -f null - 2>&1 |
    sed -n 's/.*lavfi\.freezedetect\.freeze_\(start\|duration\): \(.*\)/\1 \2/p' |
    awk -v total="$gif_total_s" '
      $1 == "start" { start = $2; have = 1; next }
      $1 == "duration" && have { printf "  %8.1fs  ->%8.1fs  (%.1fs)\n", start, start + $2, $2; have = 0; last = start + $2; next }
      END {
        if (have) printf "  %8.1fs  ->%8.1fs  (%.1fs)  <- runs to the end\n", start, total, total - start
      }'
  exit 0
fi

if (( ${#gif_args[@]} == 2 )); then
  gif_output="${gif_args[1]}"
else
  gif_output="${gif_input%.*}-${gif_speed}x.gif"
fi
if [[ -e "$gif_output" ]] && (( ! gif_force )); then
  echo "Output already exists: $gif_output (pass --force to overwrite)" >&2
  exit 1
fi

# Trim with an input seek plus an explicit duration: -ss/-t on the input avoid
# decoding the parts we are throwing away.
gif_trim_args=(-ss "$(gif_seconds "$gif_start")")
if [[ -n "$gif_end" ]]; then
  gif_span="$(awk -v a="$(gif_seconds "$gif_start")" -v b="$(gif_seconds "$gif_end")" 'BEGIN { printf "%.3f", b - a }')"
  awk -v d="$gif_span" 'BEGIN { exit !(d > 0) }' || { echo "--end must be after --start" >&2; exit 1; }
  gif_trim_args+=(-t "$gif_span")
fi

gif_filter="setpts=(PTS-STARTPTS)/$gif_speed,fps=$gif_fps,scale=$gif_width:-1:flags=lanczos,setsar=1"
# A screen recording holds the same picture for seconds at a time, but h264
# noise makes those frames differ by a pixel here and there, so the GIF encoder
# cannot collapse them. mpdecimate drops the near-duplicates and -fps_mode vfr
# turns each survivor into one long-delay frame.
if (( gif_decimate )); then
  gif_filter="$gif_filter,mpdecimate=hi=64*12:lo=64*5:frac=0.05"
  gif_frame_mode=vfr
else
  gif_frame_mode=passthrough
fi

case "$gif_dither" in
  bayer) gif_paletteuse='paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle' ;;
  none) gif_paletteuse='paletteuse=dither=none:diff_mode=rectangle' ;;
  *) echo "Unknown --dither: $gif_dither (want bayer or none)" >&2; exit 1 ;;
esac

gif_work_dir="$(mktemp -d "${TMPDIR:-/tmp}/farallon-gif.XXXXXX")"
gif_frames="$gif_work_dir/frames.mkv"
gif_palette="$gif_work_dir/palette.png"
trap 'rm -rf -- "$gif_work_dir"' EXIT

# Pass 1: decode the source once into the frames that survive decimation, so
# the palette and encode passes never touch the multi-gigabyte original again.
"$gif_ffmpeg" -n -hide_banner -loglevel error "${gif_trim_args[@]}" -i "$gif_input" \
  -vf "$gif_filter" -fps_mode "$gif_frame_mode" -an -c:v ffv1 "$gif_frames"

# Pass 2: choose a global palette, weighting pixels that change between frames.
"$gif_ffmpeg" -n -hide_banner -loglevel error -i "$gif_frames" \
  -vf 'palettegen=stats_mode=diff' -frames:v 1 -update 1 "$gif_palette"

# Pass 3: map to the palette and encode only the changed rectangles.
"$gif_ffmpeg" -y -hide_banner -loglevel error -i "$gif_frames" -i "$gif_palette" \
  -filter_complex "[0:v][1:v]$gif_paletteuse" \
  -an -loop 0 -gifflags +transdiff -fps_mode passthrough \
  -final_delay "${GIF_FINAL_DELAY:--1}" "$gif_output"

printf 'Created: %s (%s)\n' "$gif_output" "$(wc -c <"$gif_output" | awk '{ printf "%.1f MB", $1 / 1e6 }')"
