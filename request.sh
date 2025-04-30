#!/bin/bash

# Usage: ./request.sh <URL> <DATA>

URL="$1"
DATA="$2"

if [ -z "$URL" ] || [ -z "$DATA" ]; then
  echo "Usage: $0 <URL> <JSON_DATA>"
  exit 1
fi

curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$DATA"