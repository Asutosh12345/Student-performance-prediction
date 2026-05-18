import csv
import os
import sys
from pathlib import Path

# create a sample CSV file from the first 3 rows of student_performance_dataset (2).csv
source = Path('student_performance_dataset (2).csv')
if not source.exists():
    raise SystemExit('Source CSV not found')

sample = Path('verify_sample.csv')
with open(source, 'r', newline='', encoding='utf-8') as src, open(sample, 'w', newline='', encoding='utf-8') as dst:
    reader = csv.reader(src)
    writer = csv.writer(dst)
    for i, row in enumerate(reader):
        writer.writerow(row)
        if i >= 3:
            break

# use requests if available, else urllib
try:
    import requests
    files = {'file': ('verify_sample.csv', open(sample, 'rb'), 'text/csv')}
    resp = requests.post('http://127.0.0.1:5000/api/batch-predict', files=files)
    print('status_code:', resp.status_code)
    print(resp.text)
except ImportError:
    import urllib.request
    import urllib.parse
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    data = []
    data.append('--' + boundary)
    data.append('Content-Disposition: form-data; name="file"; filename="verify_sample.csv"')
    data.append('Content-Type: text/csv')
    data.append('')
    with open(sample, 'rb') as f:
        filedata = f.read()
    data = '\r\n'.join(data).encode('utf-8') + b'\r\n' + filedata + b'\r\n' + ('--' + boundary + '--\r\n').encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:5000/api/batch-predict', data=data)
    req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)
    with urllib.request.urlopen(req) as resp:
        print('status_code:', resp.getcode())
        print(resp.read().decode('utf-8'))
