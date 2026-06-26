#!/usr/bin/env python3
"""DP Service — privacy budget tracking microservice."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, math

PRESETS = {"DP_STRONG": 1.5, "DP_MODERATE": 0.8, "DP_RELAXED": 0.5}

class DPHandler(BaseHTTPRequestHandler):
    steps = 0
    def _epsilon(self, sigma, steps):
        q, delta = 0.01, 1e-5
        return round(q * math.sqrt(2 * max(1, steps) * math.log(1/delta)) / sigma, 4)
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        elif self.path == '/budget':
            r = {n: {'epsilon': self._epsilon(s, self.steps), 'sigma': s, 'steps': self.steps} for n, s in PRESETS.items()}
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(r).encode())
        elif self.path == '/presets':
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(PRESETS).encode())
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == '/step':
            DPHandler.steps += body.get('steps', 1)
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'steps': DPHandler.steps}).encode())
        elif self.path == '/compute':
            sigma = body.get('sigma', 1.5)
            steps = body.get('steps', 100)
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'epsilon': self._epsilon(sigma, steps), 'sigma': sigma, 'steps': steps}).encode())
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8085'))
    print(f'DP Service listening on :{port}')
    HTTPServer(('0.0.0.0', port), DPHandler).serve_forever()
