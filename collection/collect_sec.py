#!/usr/bin/env python3
"""Extract lease terminations from public SEC ABS-EE filings.

Raw XML is streamed and discarded. Set SEC_USER_AGENT to an identifying user
agent with a contact address before running this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import requests
from lxml import etree


FIELDS = (
    "assetNumber", "reportingPeriodEndDate", "originationDate",
    "scheduledTerminationDate", "acquisitionCost", "contractResidualValue",
    "baseResidualValue", "vehicleValueAmount", "vehicleManufacturerName",
    "vehicleModelName", "vehicleModelYear", "vehicleTypeCode",
    "originalLeaseTermNumber", "reportingPeriodScheduledPaymentAmount",
    "terminationIndicator", "zeroBalanceCode", "zeroBalanceEffectiveDate",
    "liquidationProceedsAmount",
    "excessFeeAmount", "chargedOffAmount",
)


def get(session: requests.Session, url: str) -> requests.Response:
    for attempt in range(5):
        try:
            response = session.get(url, timeout=120)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    raise RuntimeError("unreachable")


def filings(session: requests.Session, cik: str, start: str, end: str):
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    columns = get(session, url).json()["filings"]["recent"]
    selected = [
        (columns["filingDate"][i], columns["accessionNumber"][i])
        for i, form in enumerate(columns["form"])
        if form == "ABS-EE" and start <= columns["filingDate"][i] <= end
    ]
    return sorted(selected)


def asset_url(session: requests.Session, cik: str, accession: str) -> str:
    root = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    index = get(session, root + accession + "-index.htm").text
    candidates = re.findall(
        r'href="([^"]+\.xml)"[^<]*</a>\s*</td>\s*<td[^>]*>\s*EX-102\s*</td>',
        index, flags=re.I)
    if len(candidates) != 1:
        raise ValueError(f"Expected one EX-102 XML for {accession}; got {candidates}")
    return urljoin("https://www.sec.gov", candidates[0])


def extract(xml: bytes, filing_date: str, cik: str, accession: str):
    count = 0
    kept = 0
    for _, node in etree.iterparse(BytesIO(xml), events=("end",), tag="{*}assets"):
        count += 1
        values = {etree.QName(child).localname: child.text for child in node}
        termination = values.get("terminationIndicator") or ""
        liquidation = float(values.get("liquidationProceedsAmount") or 0)
        if termination or liquidation:
            kept += 1
            row = {field: values.get(field) for field in FIELDS}
            row.update({"cik": cik, "filing_date": filing_date, "accession": accession})
            yield row
        node.clear()
        while node.getprevious() is not None:
            del node.getparent()[0]
    print(f"  parsed={count:,} event_rows={kept:,}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        parser.error("Set SEC_USER_AGENT to an identifying user agent with contact address")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    selected = filings(session, args.cik, args.start, args.end)
    if args.limit:
        selected = selected[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = []
    with args.output.open("w") as out:
        for filing_date, accession in selected:
            url = asset_url(session, args.cik, accession)
            print(f"{filing_date} {accession} {url}", flush=True)
            response = get(session, url)
            n = 0
            for row in extract(response.content, filing_date, args.cik, accession):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            meta.append({"filing_date": filing_date, "accession": accession,
                         "xml_url": url, "bytes": len(response.content), "event_rows": n})
            time.sleep(0.3)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {args.output} ({sum(x['event_rows'] for x in meta):,} rows)", flush=True)


if __name__ == "__main__":
    main()
