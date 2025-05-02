#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_help() {
  echo -e "${YELLOW}Usage:${NC}"
  echo "  ./compress-central-hub.sh [compression-level] [--dry-run] [--no-tests]"
  echo
  echo "Arguments:"
  echo "  compression-level   1 to 5:"
  echo "    1 = No compression (default Repomix)"
  echo "    2 = Remove empty lines"
  echo "    3 = Remove empty lines + remove comments"
  echo "    4 = Compress essential signatures only (uses --compress)"
  echo "    5 = Maximum compression (compress + remove comments + remove empty lines)"
  echo
  echo "  --dry-run            Simulate without generating the output file"
  echo "  --no-tests           Exclude tests/ and stress/ directories from output"
  echo "  -h, --help           Show this help message"
}

# Default values
COMPRESSION_LEVEL=2
DRY_RUN=false
EXCLUDE_TESTS=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    -h|--help) show_help; exit 0 ;;
    --dry-run) DRY_RUN=true ;;
    --no-tests) EXCLUDE_TESTS=true ;;
    [1-5]) COMPRESSION_LEVEL=$arg ;;
    *) echo -e "${RED}❌ Unknown argument: $arg${NC}"; show_help; exit 1 ;;
  esac
done

REPO_OPTIONS="--style xml -i **/*.log,**/*.json,**/.gitignore,node_modules/**,thumbnails/**,**/dist/**,**/build/** --token-count-encoding o200k_base"

case $COMPRESSION_LEVEL in
  2) REPO_OPTIONS="--remove-empty-lines $REPO_OPTIONS" ;;
  3) REPO_OPTIONS="--remove-empty-lines --remove-comments $REPO_OPTIONS" ;;
  4) REPO_OPTIONS="--compress $REPO_OPTIONS" ;;
  5) REPO_OPTIONS="--compress --remove-empty-lines --remove-comments $REPO_OPTIONS" ;;
esac

TARGET_PATHS="src python"
[ "$EXCLUDE_TESTS" = false ] && TARGET_PATHS="$TARGET_PATHS tests stress"

if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}🧪 Dry Run Mode: Simulating Repomix...${NC}"
  echo "→ Compression level: $COMPRESSION_LEVEL"
  echo "→ Repomix options: $REPO_OPTIONS"
  echo "→ Processing: $TARGET_PATHS"
  echo "→ Output: my-central-hub.xml"
  exit 0
fi

echo -e "${YELLOW}🗑️  Deleting previous my-central-hub.xml...${NC}"
rm -f my-central-hub.xml

echo -e "${YELLOW}⚙️  Running Repomix...${NC}"
repomix $REPO_OPTIONS -o my-central-hub.xml $TARGET_PATHS

echo -e "${GREEN}✅ my-central-hub.xml has been created and is ready for LLM ingestion.${NC}"