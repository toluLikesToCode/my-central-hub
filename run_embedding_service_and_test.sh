#!/usr/bin/env bash
###############################################################################
# run_embedding_service_and_test.sh – clean, user‑friendly builder/runner
###############################################################################

set -Eeuo pipefail

# ─────────── Styles ─────────── #
RESET='\033[0m'; BOLD='\033[1m'
CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; MAGENTA='\033[35m'; DIM='\033[2m'
SPIN=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)

PRINT_INTERVAL=3
LOG_DIR="./build-logs"; mkdir -p "$LOG_DIR"
BUILD_LOG="$LOG_DIR/$(date +%F_%H-%M-%S).log"
SUBLOG=$(mktemp)
SPID=""

# ─────────── Spinner ─────────── #
start_spin() {             # $1 message
  local msg="$1"
  (
    trap '' TERM            # suppress “Terminated” echo
    tput civis
    local i=0
    while :; do
      printf "\r${BOLD}${MAGENTA}${SPIN[i]}${RESET} ${DIM}%s${RESET}" "$msg"
      i=$(( (i+1) % ${#SPIN[@]} )); sleep 0.08
    done
  ) &
  SPID=$!
}
stop_spin() {
  [[ -n "$SPID" ]] || return
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
  printf "\r%${COLUMNS:-$(tput cols)}s\r" ""
  tput cnorm
  SPID=""
}

# ─────────── Pretty helpers ─────────── #
step() { echo -e "\n${BOLD}${CYAN}╭─[ $1 ]──────────────────────────────────────────────────────╮\n${BOLD}${CYAN}╰──────────────────────────────────────────────────────────────╯${RESET}"; }
succ(){ echo -e "${GREEN}${BOLD}✔ $1${RESET}"; }
warn(){ echo -e "${YELLOW}${BOLD}⚠ $1${RESET}"; }
die() { echo -e "${RED}${BOLD}✖ $1${RESET}"; exit 1; }

log_line(){ echo -e "$1" >>"$SUBLOG"; }

show_window(){
  [[ -s "$SUBLOG" ]] || return
  echo -e "${DIM}───────────────────────────────────────────────────────────────${RESET}"
  tail -n 5 "$SUBLOG" | while read -r l; do printf "${DIM}    %s${RESET}\n" "$l"; done
  echo -e "${DIM}───────────────────────────────────────────────────────────────${RESET}"
}

# ─────────── Cleanup & error traps ─────────── #
cleanup(){ stop_spin; rm -f "$SUBLOG"; }
trap cleanup EXIT

trap 'stop_spin; echo -e "\n${RED}${BOLD}Interrupted${RESET}"; exit 130' INT
error_report(){
  local c=$?
  stop_spin
  echo -e "\n${RED}${BOLD}✖ Exit $c${RESET}" >&2
  show_window >&2
  echo -e "${RED}Trace:${RESET}" >&2; local i=0; while caller $i; do ((i++)); done >&2
  exit "$c"
}
trap error_report ERR

# ─────────── Defaults ─────────── #
IMG="myhub/clip-embed:latest"; CNT="clip-embedding-service"
HPORT=3456; CPORT=8000
VHOST="./public"; VCONT="/app/public"
DOCKERFILE="python/services/embedding-service/Dockerfile.embed"
PROJECT="."
WAIT=90; DO_TEST=0; VERB=0
TEST="tests/embeddings/test_embedding_service.py"
BOPTS=(); ROPTS=(); IN_BUILD=1

# ─────────── Arg parsing ─────────── #
while [[ $# -gt 0 ]]; do case "$1" in
  -i|--image) IMG="$2"; shift 2;;
  -c|--container) CNT="$2"; shift 2;;
  -p|--host-port) HPORT="$2"; shift 2;;
  -P|--container-port) CPORT="$2"; shift 2;;
  -v|--volume) VHOST="$2"; shift 2;;
  -d|--dockerfile) DOCKERFILE="$2"; shift 2;;
  -t|--test-script) TEST="$2"; shift 2;;
  -w|--wait) WAIT="$2"; shift 2;;
  -r|--project-root) PROJECT="$2"; shift 2;;
  --verbose) VERB=1; shift;;
  --test) DO_TEST=1; shift;;
  --) IN_BUILD=0; shift;;
  *) (( IN_BUILD )) && BOPTS+=("$1") || ROPTS+=("$1"); shift;;
esac; done

# ─────────── Config summary ─────────── #
echo -e "${BOLD}${CYAN}\n╭────────────────────[ CONFIG ]────────────────────╮${RESET}"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Image" "$IMG"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Container" "$CNT"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s:%s\n" "Ports" "$HPORT" "$CPORT"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s → %s\n" "Volume" "$VHOST" "$VCONT"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Dockerfile" "$DOCKERFILE"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Build opts" "${BOPTS[*]:-(none)}"
printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Run opts" "${ROPTS[*]:-(none)}"
[[ $DO_TEST -eq 1 ]] && printf "${BOLD}${CYAN}│${RESET} %-18s: %s\n" "Test" "$TEST"
echo -e "${BOLD}${CYAN}╰──────────────────────────────────────────────────╯${RESET}"

# ─────────── Step 1: Validate ─────────── #
step "Validate"
[[ -f "$DOCKERFILE" ]] || die "Missing Dockerfile"
[[ -d "$PROJECT"   ]] || die "Missing project dir"
succ "Dockerfile & project dir OK"

# ─────────── Step 2: Build image ───────── #
step "Build image $IMG"
start_spin "Building…"
last_print=$(date +%s)

build_stream(){
  if (( VERB )); then cat; else
    # Filter interesting lines; drop duplicates
    grep -E '(^#\d+ \[|^Step|CACHED|DONE|ERROR|error|failed|denied|not found|no such)' | awk '!seen[$0]++'
  fi
}

docker build --progress=plain -f "$DOCKERFILE" -t "$IMG" \
  "${BOPTS[@]}" "$PROJECT" 2>&1 | tee "$BUILD_LOG" | build_stream | \
  while read -r line; do
    log_line "$line"
    now=$(date +%s)
    if (( now - last_print >= PRINT_INTERVAL )); then
      last_print=$now; stop_spin; show_window; start_spin "Building…"
    fi
done
BS=${PIPESTATUS[0]}
stop_spin; show_window
(( BS == 0 )) || die "Build failed (see $BUILD_LOG)"
succ "Image built (log: $BUILD_LOG)"

# ─────────── Step 3: Clean old container ── #
step "Remove previous container"
docker rm -f "$CNT" >/dev/null 2>&1 && warn "Old container removed." || true
succ "Cleanup done"

# ─────────── Step 4: Run container ─────── #
step "Run container"
CID=$(docker run -d --name "$CNT" -p "$HPORT:$CPORT" \
      -v "$VHOST:$VCONT" "${ROPTS[@]}" "$IMG") || die "docker run failed"
succ "Running – CID=${CID::12}"
if (( WAIT > 0 )); then start_spin "Waiting $WAIT s…"; sleep "$WAIT"; stop_spin; succ "Wait done"; fi

# ─────────── Step 5: Tests (optional) ──── #
step "Tests"
if (( DO_TEST )); then
  case "$TEST" in
    *.py) python "$TEST";;
    *.ts) npx ts-node "$TEST";;
    *.js) node "$TEST";;
    *.sh) bash "$TEST";;
    *) "$TEST";;
  esac
  succ "Tests passed"
else
  warn "Tests skipped"
fi

# ─────────── Finished ─────────── #
echo -e "\n${BOLD}${CYAN}╭─────────────────────────────────────────╮\n│ Script finished successfully.\n${BOLD}${CYAN}╰─────────────────────────────────────────╯${RESET}"
echo -e "${YELLOW}Stop the container with:${RESET} ${BOLD}docker stop $CNT${RESET}"
