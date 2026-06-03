#!/usr/bin/env bash
# Tail recent Edge Function console.log output via the Supabase Management API.
# Token is read from the Supabase CLI keychain entry (no secrets stored here).
set -euo pipefail

REF="etxhwliwhwacyigsevth"
RAW="$(security find-generic-password -s 'Supabase CLI' -w 2>/dev/null)"
PAT="${RAW#go-keyring-base64:}"
case "$RAW" in
  go-keyring-base64:*) PAT="$(printf '%s' "$PAT" | base64 --decode)";;
esac

curl -s -G "https://api.supabase.com/v1/projects/$REF/analytics/endpoints/logs.all" \
  -H "Authorization: Bearer $PAT" \
  --data-urlencode "sql=select t.timestamp, m.level, t.event_message from function_logs t cross join unnest(t.metadata) as m order by t.timestamp desc limit 60" \
  | python3 -c "
import sys, json, datetime
d = json.load(sys.stdin)
if d.get('error'):
    print('ERROR:', d['error']); sys.exit(1)
rows = d.get('result', [])
if not rows:
    print('(no console logs in the last hour — send the bot a voice note, then re-run)')
for r in reversed(rows):
    ts = r.get('timestamp')
    try:
        ts = datetime.datetime.fromtimestamp(int(ts)/1_000_000).strftime('%H:%M:%S')
    except Exception:
        pass
    print(f\"{ts} [{r.get('level','')}] {r.get('event_message','').rstrip()}\")
"
