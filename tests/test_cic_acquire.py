import sys, os, tempfile, json, hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

td = tempfile.mkdtemp()
src = Path(td) / 'source'
src.mkdir()
out = Path(td) / 'output'

csv_content = 'Flow ID,Src IP,Dst IP,Label\n1,10.0.0.1,10.0.0.2,BENIGN\n2,10.0.0.2,10.0.0.3,Attack\n'
fake_csv = src / 'Monday-WorkingHours.pcap_ISCX.csv'
fake_csv.write_bytes(csv_content.encode('utf-8'))

# Compute expected values from the actual file bytes
expected_sha = hashlib.sha256(fake_csv.read_bytes()).hexdigest()
expected_size = fake_csv.stat().st_size

manifest = {
  'dataset_id': 'cic-ids2017',
  'csv_files': [{
    'name': 'Monday-WorkingHours.pcap_ISCX.csv',
    'day': 'Monday',
    'description': 'Benign baseline',
    'sha256': None,
    'sha256_verified': False,
    'size_bytes': None,
    'size_verified': False,
    'record_count': None,
    'verified': False
  }],
  'verification': {'checksums_recorded': False}
}
manifest_path = Path(td) / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))

from scripts.data.acquire_cic_ids2017 import (
    load_manifest, build_file_records, verify_record,
    update_manifest_from_records, save_manifest, copy_to_output
)

m = load_manifest(manifest_path)
records = build_file_records(m, [src], out)
results = [verify_record(r, out) for r in records]
r = results[0]
assert r['checksum_computed'] == expected_sha, 'checksum mismatch'
assert r['checksum_match'] is None  # no expected checksum in manifest
assert r['size_bytes'] == expected_size, f'size mismatch: {r["size_bytes"]} != {expected_size}'
assert r['record_count'] == 2
assert not r['errors'], r['errors']

err = copy_to_output(records[0])
assert err is None, 'copy error: ' + str(err)
assert (out / 'Monday-WorkingHours.pcap_ISCX.csv').is_file()

m2 = update_manifest_from_records(m, records, results)
assert m2['csv_files'][0]['sha256'] == expected_sha
assert m2['csv_files'][0]['sha256_verified'] is True
assert m2['csv_files'][0]['verified'] is True
assert m2['verification']['checksums_recorded'] is True

print('ACQUISITION SCRIPT FUNCTIONAL TEST PASSED')
print('  checksum:', expected_sha[:16] + '...')
print('  records:', r['record_count'])
print('  size:', r['size_bytes'], 'bytes')
