# -*- coding: utf-8 -*-
"""DZ PhotoID Studio - local launcher server.

Serves this folder on the first free port from 8765 upward and opens
the browser automatically. Pure Python standard library: nothing to
install beyond Python 3 itself.
"""
import http.server
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST_PORT = 8765
LAST_PORT = 8785


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console window clean


def main():
    handler = partial(QuietHandler, directory=str(ROOT))
    server = None
    port = None
    for candidate in range(FIRST_PORT, LAST_PORT + 1):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print("[Error] All ports %d-%d are busy." % (FIRST_PORT, LAST_PORT))
        print("Close the program using one of them and try again.")
        input("Press Enter to exit...")
        return 1
    url = "http://localhost:%d" % port
    print("Server is running at: %s" % url)
    print("To stop the program: close this window or press Ctrl+C")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
