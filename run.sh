#!/usr/bin/env bash
# DZ PhotoID Studio — مشغّل لينكس / ماك
cd "$(dirname "$0")"
PORT=8765
echo "============================================================"
echo "  DZ PhotoID Studio — صانع صور الهوية — الجزائر"
echo "  http://localhost:$PORT"
echo "  للإيقاف: Ctrl+C"
echo "============================================================"
(python3 -c "import webbrowser,time; time.sleep(1.5); webbrowser.open('http://localhost:$PORT')" &)
python3 -m http.server $PORT --bind 127.0.0.1
