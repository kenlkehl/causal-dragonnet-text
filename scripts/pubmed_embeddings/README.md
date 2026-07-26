# PubMed External Embedding Corpus

These scripts build a cancer-related PubMed title/abstract corpus and embed it
into the external chunk-cache format used by multi-model agentic forest
embedding contrast retrieval.

Choose the corpus query, ordering, and record bound explicitly. For example:

```bash
python scripts/pubmed_embeddings/download_pubmed_cancer.py \
  --output-dir /absolute/path/to/pubmed_embeddings \
  --max-records 100000 \
  --query 'cancer[Title/Abstract] AND hasabstract[text]' \
  --sort relevance \
  --email your_email@example.org
```

Embed the downloaded JSONL into a resumable external cache:

```bash
python scripts/pubmed_embeddings/embed_pubmed_corpus.py \
  --input /absolute/path/to/pubmed_embeddings/pubmed_cancer_abstracts.jsonl \
  --output-cache-dir /absolute/path/to/pubmed_embeddings/pubmed_cancer_embedding_cache \
  --model-name /absolute/path/to/embedding-model \
  --corpus-name pubmed_cancer \
  --text-column text \
  --source-id-column pmid \
  --max-seq-length 1024 \
  --chunk-size-words 256 \
  --chunk-overlap-words 64 \
  --max-chunks 32 \
  --chunk-selection first \
  --normalize-embeddings \
  --device-ids 0 1 \
  --batch-size 32
```

The numbers above are examples, not source defaults. `max-chunks` is an
abort-only allocation bound. If either word chunking or tokenizer-aware
rechunking requires more chunks, the build fails without selecting the first or
last chunks; increase the configured bound and start a scientifically distinct
cache build.

If interrupted, rerun the same embedding command. Completed part directories
under `_parts/` are reused, and a partially encoded part resumes from its last
saved chunk offset. After all parts finish, the script merges them into
`chunk_embeddings.npy`, `offsets.npy`, `chunk_texts.jsonl`, `row_metadata.jsonl`,
and `metadata.json`.

Use the result in the oracle multi-model path with:

```bash
--embedding-external-cache-dir /absolute/path/to/pubmed_embeddings/pubmed_cancer_embedding_cache
```
