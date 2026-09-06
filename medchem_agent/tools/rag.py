"""Literature RAG: PubMed E-utilities retrieval + BM25 re-ranking.

Connects directly (trust_env=False) so requests bypass any system proxy.
Results are cached per query to disk to stay polite to NCBI.
"""
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import httpx
from rank_bm25 import BM25Okapi

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _parse_abstracts(xml_text: str):
    root = ET.fromstring(xml_text)
    docs = []
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = art.findtext(".//ArticleTitle") or ""
        ab_elems = art.findall(".//AbstractText")
        abstract = " ".join("".join(t.itertext()) for t in ab_elems)
        docs.append({"pmid": pmid, "title": title.strip(), "abstract": abstract.strip()})
    return docs


class PubMedRAG:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.client = httpx.Client(trust_env=False, timeout=30)

    def _cache_path(self, query: str):
        h = hashlib.md5(query.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{h}.json")

    def retrieve(self, query: str, top_k: int = 10, retmax: int = 40):
        """Return a list of {pmid, title, abstract} ranked by BM25 against `query`."""
        cache = self._cache_path(query)
        if os.path.exists(cache):
            with open(cache) as f:
                docs = json.load(f)
        else:
            docs = self._search_and_fetch(query, retmax=retmax)
            with open(cache, "w") as f:
                json.dump(docs, f)
        if not docs:
            return []

        corpus = [_tokenize(d["title"] + " " + d["abstract"]) for d in docs]
        bm25 = BM25Okapi(corpus)
        q_tokens = _tokenize(query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in ranked]

    def _search_and_fetch(self, query: str, retmax: int = 40):
        # esearch
        r = self.client.get(f"{EUTILS}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": retmax, "retmode": "json",
        })
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        time.sleep(0.4)  # NCBI politeness
        # efetch
        r = self.client.get(f"{EUTILS}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml",
        })
        r.raise_for_status()
        return _parse_abstracts(r.text)

    def close(self):
        self.client.close()
