#!/usr/bin/env python3
"""PSA Service — entity alignment microservice."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, sys, os
sys.path.insert(0, '/app')
from fl_pets.psa import align_entities_fuzzy, align_entities_exact

class PSAHandler(BaseHTTPRequestHandler):
    parties_data = {}
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        elif self.path == '/status':
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'parties': list(self.parties_data.keys()), 'ready': len(self.parties_data) >= 2}).encode())
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == '/register':
            pid = body.get('party_id')
            PSAHandler.parties_data[pid] = body.get('records', [])
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'party_id': pid, 'records': len(PSAHandler.parties_data[pid])}).encode())
        elif self.path == '/align_exact':
            result = align_entities_exact(PSAHandler.parties_data, salt=os.urandom(32))
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == '/align_fuzzy':
            t = body.get('threshold', 0.7)
            result = align_entities_fuzzy(PSAHandler.parties_data, threshold=t)
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8084'))
    print(f'PSA Service listening on :{port}')
    HTTPServer(('0.0.0.0', port), PSAHandler).serve_forever()
