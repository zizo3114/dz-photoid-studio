#!/usr/bin/env bash
# DZ PhotoID Studio - local launcher (Linux / macOS)
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 serve.py
elif command -v python >/dev/null 2>&1; then
  python serve.py
else
  echo "Python 3 is not installed. Please install it from https://www.python.org"
  read -r -p "Press Enter to exit..."
fi
