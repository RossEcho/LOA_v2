#!/data/data/com.termux/files/usr/bin/bash
TARGET="$1"
COUNT="${2:-4}"

if [ -z "$TARGET" ]; then
  echo '{"error":"missing target"}'
  exit 1
fi

OUT=$(ping -c "$COUNT" "$TARGET" 2>&1)

echo "{
  \"tool\": \"ping\",
  \"target\": \"$TARGET\",
  \"output\": $(printf '%s\n' "$OUT" | jq -Rs .)
}"
