#!/usr/bin/env python3
"""Audit Logger — immutable exchange log microservice."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, time, hashlib

LOG_DIR = os.environ.get('AUDIT_DIR', '/app/audit_logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'audit.jsonl')

class AuditHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        elif self.path == '/logs':
            entries = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    entries = [json.loads(line) for line in f]
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(entries).encode())
        elif self.path == '/count':
            count = 0
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    count = sum(1 for _ in f)
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'count': count}).encode())
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path == '/log':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            entry = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'source': body.get('source', 'unknown'),
                'destination': body.get('destination', 'unknown'),
                'action': body.get('action', 'unknown'),
                'round': body.get('round', 0),
                'payload_hash': body.get('payload_hash', ''),
                'metadata': body.get('metadata', {}),
            }
            entry['entry_hash'] = hashlib.sha256(json.dumps(entry).encode()).hexdigest()[:16]
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            self.send_response(201); self.end_headers()
            self.wfile.write(json.dumps(entry).encode())
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8086'))
    print(f'Audit Logger listening on :{port}')
    HTTPServer(('0.0.0.0', port), AuditHandler).serve_forever()
