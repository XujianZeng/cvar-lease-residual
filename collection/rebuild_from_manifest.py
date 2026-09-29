"""Recreate the exact filing selection in filings_used.csv; no discovery query.

Set SEC_USER_AGENT to your own identifying name and contact address. Existing
dataset files are not overwritten. No requests are made without --download.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import requests
from .collect_sec import extract, get


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).parent.parent
    groups = defaultdict(list)
    with (root / 'filings_used.csv').open() as handle:
        for row in csv.DictReader(handle):
            groups[row['dataset']].append(row)
    print(f'{len(groups)} datasets; {sum(map(len, groups.values()))} filings')
    if not args.download:
        return
    agent = os.environ.get('SEC_USER_AGENT')
    if not agent:
        parser.error('Set SEC_USER_AGENT to your identifying name and contact address')
    data = root / 'data'
    data.mkdir(exist_ok=True)
    targets = [data / (name + '.jsonl') for name in groups]
    if any(p.exists() for p in targets):
        parser.error('One or more target files already exist; use a fresh directory')
    session = requests.Session()
    session.headers.update({'User-Agent': agent, 'Accept-Encoding': 'gzip, deflate'})
    for name, filings in groups.items():
        target = data / (name + '.jsonl')
        partial = target.with_suffix('.jsonl.partial')
        with partial.open('x') as handle:
            for filing in filings:
                url = filing['xml_url']
                # Accession prefixes may identify a filing agent, not the trust.
                cik = url.split('/data/', 1)[1].split('/', 1)[0].zfill(10)
                response = get(session, url)
                count = 0
                for row in extract(response.content, filing['filing_date'], cik, filing['accession']):
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                if count != int(filing['event_rows']):
                    raise ValueError(f'Event-row count differs for {url}: {count}')
                time.sleep(0.3)
        partial.rename(target)
        print(f'Saved {target.name}')


if __name__ == '__main__':
    main()
