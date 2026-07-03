#!/usr/bin/env python3
"""Download cancer-related PubMed titles and abstracts to JSONL.

The output JSONL is intentionally simple so it can be embedded by
``embed_pubmed_corpus.py`` and used as an external corpus for embedding contrast
retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_OUTPUT_ROOT = Path("/data1/ken/pcori_dev/pubmed_embeddings")
DEFAULT_OUTPUT_NAME = "pubmed_cancer_abstracts.jsonl"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = (
    "("
    "cancer[Title/Abstract] OR cancers[Title/Abstract] OR "
    "tumor[Title/Abstract] OR tumors[Title/Abstract] OR "
    "tumour[Title/Abstract] OR tumours[Title/Abstract] OR "
    "neoplasm[Title/Abstract] OR neoplasms[Title/Abstract] OR "
    "neoplasms[MeSH Terms] OR malignan*[Title/Abstract] OR "
    "oncology[Title/Abstract] OR oncologic[Title/Abstract] OR "
    "leukemia[Title/Abstract] OR leukaemia[Title/Abstract] OR "
    "lymphoma[Title/Abstract] OR carcinoma[Title/Abstract] OR "
    "adenocarcinoma[Title/Abstract] OR melanoma[Title/Abstract] OR "
    "sarcoma[Title/Abstract] OR glioma[Title/Abstract] OR "
    "myeloma[Title/Abstract] OR blastoma[Title/Abstract]"
    ") AND hasabstract[text]"
)


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download cancer-related PubMed titles and abstracts."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--sort", default="relevance")
    parser.add_argument("--email", default=None, help="NCBI contact email.")
    parser.add_argument("--api-key", default=None, help="Optional NCBI API key.")
    parser.add_argument(
        "--tool",
        default="causal-dragonnet-text-pubmed-downloader",
        help="NCBI E-utilities tool name.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Delay between E-utilities calls. Defaults to NCBI-friendly limits.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    checkpoint_path = (
        Path(args.checkpoint_path).expanduser()
        if args.checkpoint_path
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )
    download_pubmed(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        query=args.query,
        max_records=args.max_records,
        batch_size=args.batch_size,
        sort=args.sort,
        email=args.email,
        api_key=args.api_key,
        tool=args.tool,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
    )
    print(output_path)


def download_pubmed(
    *,
    output_path: Path,
    checkpoint_path: Path,
    query: str,
    max_records: int,
    batch_size: int,
    sort: str,
    email: Optional[str],
    api_key: Optional[str],
    tool: str,
    sleep_seconds: Optional[float],
    force: bool,
) -> None:
    if max_records < 1:
        raise ValueError("--max-records must be >= 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if force:
        _unlink_if_exists(output_path)
        _unlink_if_exists(checkpoint_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pmids = _load_existing_pmids(output_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    retstart = int(checkpoint.get("retstart", len(seen_pmids)) or 0)
    written = len(seen_pmids)
    delay = sleep_seconds if sleep_seconds is not None else (0.11 if api_key else 0.34)
    common = _common_eutils_params(email=email, api_key=api_key, tool=tool)

    search = _esearch(
        query=query,
        max_records=max_records,
        sort=sort,
        common=common,
    )
    available = int(search["count"])
    query_key = str(search["query_key"])
    webenv = str(search["webenv"])
    logger.info(
        "PubMed search returned %d records; target=%d, existing=%d, retstart=%d",
        available,
        max_records,
        written,
        retstart,
    )

    target = min(int(max_records), available)
    with open(output_path, "a", encoding="utf-8") as out:
        while written < target and retstart < available:
            retmax = min(int(batch_size), target - written, available - retstart)
            if retmax <= 0:
                break
            xml_bytes = _efetch(
                query_key=query_key,
                webenv=webenv,
                retstart=retstart,
                retmax=retmax,
                common=common,
            )
            records = list(_parse_pubmed_xml(xml_bytes))
            if not records:
                logger.warning("No records parsed at retstart=%d; stopping", retstart)
                break

            new_records = 0
            for record in records:
                pmid = str(record.get("pmid") or "").strip()
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_records += 1
                written += 1
                if written >= target:
                    break
            out.flush()
            retstart += len(records)
            _write_checkpoint(
                checkpoint_path,
                {
                    "query": query,
                    "max_records": int(max_records),
                    "available": available,
                    "written": written,
                    "retstart": retstart,
                    "updated_at": datetime.now().isoformat(),
                    "output_path": str(output_path),
                },
            )
            logger.info(
                "Downloaded batch: parsed=%d new=%d written=%d/%d next_retstart=%d",
                len(records),
                new_records,
                written,
                target,
                retstart,
            )
            if written < target and delay > 0:
                time.sleep(delay)


def _esearch(
    *,
    query: str,
    max_records: int,
    sort: str,
    common: Dict[str, str],
) -> Dict[str, Any]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "usehistory": "y",
        "sort": sort,
        **common,
    }
    payload = _eutils_post("esearch.fcgi", params)
    root = ET.fromstring(payload)
    count = int(_text(root.find("Count")) or "0")
    query_key = _text(root.find("QueryKey"))
    webenv = _text(root.find("WebEnv"))
    if not query_key or not webenv:
        raise RuntimeError("ESearch did not return QueryKey/WebEnv history tokens")
    return {
        "count": min(count, int(max_records)),
        "query_key": query_key,
        "webenv": webenv,
    }


def _efetch(
    *,
    query_key: str,
    webenv: str,
    retstart: int,
    retmax: int,
    common: Dict[str, str],
) -> bytes:
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": int(retstart),
        "retmax": int(retmax),
        "retmode": "xml",
        **common,
    }
    return _eutils_post("efetch.fcgi", params)


def _eutils_post(endpoint: str, params: Dict[str, Any], retries: int = 4) -> bytes:
    url = f"{EUTILS_BASE}/{endpoint}"
    encoded = urllib.parse.urlencode(params).encode("utf-8")
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=encoded, method="POST")
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError as exc:
            last_exc = exc
        sleep = min(30.0, 2.0**attempt)
        logger.warning(
            "E-utilities request failed (%s); retrying in %.1fs",
            last_exc,
            sleep,
        )
        time.sleep(sleep)
    raise RuntimeError(f"E-utilities request failed after {retries} attempts") from last_exc


def _parse_pubmed_xml(xml_bytes: bytes) -> Iterable[Dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        article_node = medline.find("Article")
        if article_node is None:
            continue
        pmid = _text(medline.find("PMID"))
        title = _itertext(article_node.find("ArticleTitle"))
        abstract = _abstract_text(article_node)
        if not pmid or not title or not abstract:
            continue
        journal = _itertext(article_node.find("Journal/Title"))
        year = _publication_year(article_node)
        article_ids = _article_ids(article)
        mesh_headings = [
            _itertext(node.find("DescriptorName"))
            for node in medline.findall("MeshHeadingList/MeshHeading")
        ]
        mesh_headings = [heading for heading in mesh_headings if heading]
        publication_types = [
            _itertext(node) for node in article_node.findall("PublicationTypeList/PublicationType")
        ]
        publication_types = [kind for kind in publication_types if kind]
        record = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "text": f"{title}\n\n{abstract}",
            "journal": journal,
            "year": year,
            "doi": article_ids.get("doi"),
            "pmcid": article_ids.get("pmc"),
            "publication_types": publication_types,
            "mesh_headings": mesh_headings,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        yield record


def _abstract_text(article_node: ET.Element) -> str:
    parts: List[str] = []
    for abstract_text in article_node.findall("Abstract/AbstractText"):
        label = str(abstract_text.attrib.get("Label") or "").strip()
        text = _itertext(abstract_text)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts).strip()


def _article_ids(article: ET.Element) -> Dict[str, str]:
    ids: Dict[str, str] = {}
    for node in article.findall("PubmedData/ArticleIdList/ArticleId"):
        id_type = str(node.attrib.get("IdType") or "").strip().lower()
        value = _itertext(node)
        if id_type and value:
            ids[id_type] = value
    return ids


def _publication_year(article_node: ET.Element) -> Optional[int]:
    for path in [
        "ArticleDate/Year",
        "Journal/JournalIssue/PubDate/Year",
        "Journal/JournalIssue/PubDate/MedlineDate",
    ]:
        value = _text(article_node.find(path))
        if not value:
            continue
        digits = "".join(ch for ch in value[:4] if ch.isdigit())
        if len(digits) == 4:
            return int(digits)
    return None


def _common_eutils_params(
    *,
    email: Optional[str],
    api_key: Optional[str],
    tool: str,
) -> Dict[str, str]:
    params = {"tool": str(tool)}
    if email:
        params["email"] = str(email)
    if api_key:
        params["api_key"] = str(api_key)
    return params


def _load_existing_pmids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    pmids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(payload.get("pmid") or "").strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _itertext(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _text(node: Optional[ET.Element]) -> str:
    return _itertext(node)


if __name__ == "__main__":
    main()
