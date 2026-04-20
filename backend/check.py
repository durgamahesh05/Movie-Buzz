import json
with open('parsed_errors.json') as f:
    d = json.load(f)

for e in d.get('generalDiagnostics', []):
    if e['severity'] == 'error': # Only show errors
        print(f"{e['file']}:{e['range']['start']['line']} - {e['message']} ({e['rule']})")
