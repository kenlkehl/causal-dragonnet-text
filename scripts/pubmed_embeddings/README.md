# PubMed External Embedding Corpus

These scripts build a cancer-related PubMed title/abstract corpus and embed it
into the external chunk-cache format used by multi-model agentic forest
embedding contrast retrieval.

Download 100,000 cancer-related PubMed records:

```bash
python scripts/pubmed_embeddings/download_pubmed_cancer.py \
  --output-dir /data1/ken/pcori_dev/pubmed_embeddings \
  --max-records 100000 \
  --email your_email@example.org
```

Embed the downloaded JSONL into a resumable external cache:

```bash
python scripts/pubmed_embeddings/embed_pubmed_corpus.py \
  --input /data1/ken/pcori_dev/pubmed_embeddings/pubmed_cancer_abstracts.jsonl \
  --output-cache-dir /data1/ken/pcori_dev/pubmed_embeddings/pubmed_cancer_embedding_cache \
  --model-name Qwen/Qwen3-Embedding-8B \
  --device-ids 0 1 \
  --batch-size 32
```

If interrupted, rerun the same embedding command. Completed part directories
under `_parts/` are reused, and a partially encoded part resumes from its last
saved chunk offset. After all parts finish, the script merges them into
`chunk_embeddings.npy`, `offsets.npy`, `chunk_texts.jsonl`, `row_metadata.jsonl`,
and `metadata.json`.

Use the result in the oracle multi-model path with:

```bash
--embedding-external-cache-dir /data1/ken/pcori_dev/pubmed_embeddings/pubmed_cancer_embedding_cache
```
