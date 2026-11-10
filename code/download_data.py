"""Download the exact public dataset mirror used in the report."""
from pathlib import Path
import hashlib, json, zipfile, requests

COMMIT='844c7fe2153031240ebd7856b8054f09a58f6f86'
ROOT=Path(__file__).resolve().parent/'data'
ROOT.mkdir(exist_ok=True)
archive=ROOT/'smoke5k-main.zip'
url=f'https://codeload.github.com/MultiX-Amsterdam/smoke5k/zip/{COMMIT}'
if not archive.exists():
    response=requests.get(url,stream=True,timeout=(30,180))
    response.raise_for_status()
    part=archive.with_suffix('.part')
    with part.open('wb') as f:
        for chunk in response.iter_content(1024*1024):
            if chunk: f.write(chunk)
    part.replace(archive)
with zipfile.ZipFile(archive) as z:
    actual=z.comment.decode('ascii')
    if actual != COMMIT: raise RuntimeError(f'Archive version mismatch: {actual}')
    prefix=z.namelist()[0].split('/')[0]
    target=ROOT/'smoke5k-main'
    if not target.exists():
        # Only archive entries below the expected prefix are extracted.
        base=target.resolve()
        for item in z.infolist():
            relative=Path(item.filename).relative_to(prefix)
            destination=(base/relative).resolve()
            if destination != base and base not in destination.parents:
                raise RuntimeError('Unexpected archive path')
            if item.is_dir(): destination.mkdir(parents=True,exist_ok=True)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(z.read(item))
manifest={'repository':'https://github.com/MultiX-Amsterdam/smoke5k','commit':COMMIT,'download_url':url,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'original_paper_doi':'10.1609/aaai.v36i3.20207'}
(ROOT/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Dataset ready:',ROOT/'smoke5k-main')
