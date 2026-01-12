# cdt/models/cnn_extractor.py
"""Feature extractor using a simple 1D CNN on text."""

import logging
import re
from collections import Counter
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class WordTokenizer:
    """
    Word-level tokenizer for CNN input.

    Tokenizes text by splitting on whitespace and punctuation.
    Vocabulary is built by calling fit() on training data.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(
        self,
        max_length: int = 2048,
        min_freq: int = 2,
        max_vocab_size: Optional[int] = 50000
    ):
        """
        Initialize word tokenizer.

        Args:
            max_length: Maximum sequence length in tokens
            min_freq: Minimum frequency for a word to be included in vocabulary
            max_vocab_size: Maximum vocabulary size (None for unlimited)
        """
        self.max_length = max_length
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # Initialize with special tokens only
        self.word_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID
        }
        self.id_to_word: Dict[int, str] = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN
        }
        self._is_fitted = False

    @property
    def vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.word_to_id)

    @property
    def pad_token(self) -> int:
        """Return PAD token ID."""
        return self.PAD_ID

    @property
    def unk_token(self) -> int:
        """Return UNK token ID."""
        return self.UNK_ID

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Splits on whitespace and separates punctuation as individual tokens.
        Lowercases all tokens.
        """
        # Lowercase and normalize whitespace
        text = text.lower().strip()

        # Split on whitespace, keeping punctuation as separate tokens
        # This regex splits on spaces and keeps punctuation as separate tokens
        tokens = re.findall(r"[a-z0-9]+|[^\s\w]", text)

        return tokens

    def fit(self, texts: List[str]) -> 'WordTokenizer':
        """
        Build vocabulary from training texts.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting word tokenizer on {len(texts)} texts...")

        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Filter by minimum frequency
        filtered_words = [
            word for word, count in word_counts.items()
            if count >= self.min_freq
        ]

        # Sort by frequency (most common first) for consistent ordering
        filtered_words.sort(key=lambda w: (-word_counts[w], w))

        # Apply max vocab size limit
        if self.max_vocab_size is not None:
            # Reserve 2 slots for special tokens
            max_words = self.max_vocab_size - 2
            filtered_words = filtered_words[:max_words]

        # Build vocabulary
        self.word_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID
        }
        self.id_to_word = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN
        }

        for word in filtered_words:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self._is_fitted = True
        logger.info(f"Vocabulary built: {self.vocab_size} tokens "
                    f"(filtered from {len(word_counts)} unique words)")

        return self

    def __call__(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = 'pt'
    ) -> dict:
        """
        Tokenize texts to word IDs.

        Args:
            texts: List of text strings
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Tokenizer has not been fitted. Call fit() with training texts first."
            )

        if max_length is None:
            max_length = self.max_length

        batch_ids = []
        batch_masks = []

        for text in texts:
            # Tokenize into words
            tokens = self._tokenize(text)

            # Truncate if needed
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Convert to IDs
            ids = [
                self.word_to_id.get(token, self.UNK_ID)
                for token in tokens
            ]

            batch_ids.append(ids)

        # Pad to max length in batch
        if padding:
            max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
            for i, ids in enumerate(batch_ids):
                pad_len = max_len - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.PAD_ID] * pad_len
                batch_ids[i] = ids
                batch_masks.append(mask)
        else:
            batch_masks = [[1] * len(ids) for ids in batch_ids]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(batch_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_masks, dtype=torch.long)
            }
        return {'input_ids': batch_ids, 'attention_mask': batch_masks}

    def get_state(self) -> Dict[str, Any]:
        """Get tokenizer state for serialization."""
        return {
            'max_length': self.max_length,
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size,
            'word_to_id': self.word_to_id,
            'is_fitted': self._is_fitted
        }

    def load_state(self, state: Dict[str, Any]) -> 'WordTokenizer':
        """Load tokenizer state from serialization."""
        self.max_length = state['max_length']
        self.min_freq = state['min_freq']
        self.max_vocab_size = state['max_vocab_size']
        self.word_to_id = state['word_to_id']
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self._is_fitted = state['is_fitted']
        return self


class CNNFeatureExtractor(nn.Module):
    """
    Extract text representations using a simple 1D CNN.

    Architecture:
    1. Word embedding layer (vocabulary learned from training data)
    2. Multiple 1D CNN layers with different kernel sizes (n-gram style)
    3. Global max pooling over sequence
    4. Optional projection layer to match downstream dimension

    This is much simpler and faster than transformer models while still
    capturing local patterns in text.

    IMPORTANT: Call fit_tokenizer(texts) with training data before using
    the model for training or inference.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_filters: int = 256,
        kernel_sizes: List[int] = [3, 4, 5, 7],
        projection_dim: Optional[int] = 128,
        dropout: float = 0.1,
        max_length: int = 2048,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 50000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize CNN feature extractor.

        Args:
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes (captures different n-gram lengths)
            projection_dim: Final output dimension. If None, use num_filters * len(kernel_sizes)
            dropout: Dropout rate
            max_length: Maximum sequence length in tokens
            min_word_freq: Minimum word frequency to include in vocabulary
            max_vocab_size: Maximum vocabulary size
            device: Device to place model on
        """
        super().__init__()

        self.max_length = max_length
        self._projection_dim = projection_dim
        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout

        if device is None:
            device = torch.device('cpu')
        self._device = device

        # Word-level tokenizer (must be fitted before use)
        self.tokenizer = WordTokenizer(
            max_length=max_length,
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # Embedding layer - will be created/rebuilt when tokenizer is fitted
        # Initialize with placeholder size; will be replaced after fit_tokenizer()
        self._placeholder_vocab_size = 100
        self.embedding = nn.Embedding(
            num_embeddings=self._placeholder_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.tokenizer.pad_token
        )

        # 1D CNN layers with different kernel sizes
        # Use padding='same' to ensure output length matches input length
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Calculate CNN output dimension
        cnn_output_dim = num_filters * len(kernel_sizes)
        self.hidden_size = cnn_output_dim

        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(cnn_output_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            logger.info(f"Added projection layer: {cnn_output_dim} -> {projection_dim}")
        else:
            self.projection = None

        # Final normalization
        self.feature_norm = nn.LayerNorm(self.output_dim)

        logger.info(f"CNNFeatureExtractor initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Num filters: {num_filters}")
        logger.info(f"  Kernel sizes: {kernel_sizes}")
        logger.info(f"  CNN output dim: {cnn_output_dim}")
        logger.info(f"  Output dim: {self.output_dim}")
        logger.info(f"  NOTE: Call fit_tokenizer() before training")

    def fit_tokenizer(self, texts: List[str]) -> 'CNNFeatureExtractor':
        """
        Fit the tokenizer on training texts and rebuild the embedding layer.

        This MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        # Fit tokenizer to build vocabulary
        self.tokenizer.fit(texts)

        # Rebuild embedding layer with correct vocabulary size
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self.tokenizer.pad_token
        )

        # Move to device
        self.embedding = self.embedding.to(self._device)

        logger.info(f"Tokenizer fitted and embedding layer rebuilt:")
        logger.info(f"  Vocabulary size: {self.tokenizer.vocab_size}")

        return self

    def get_tokenizer_state(self) -> Dict[str, Any]:
        """Get tokenizer state for checkpoint saving."""
        return self.tokenizer.get_state()

    def load_tokenizer_state(self, state: Dict[str, Any]) -> 'CNNFeatureExtractor':
        """
        Load tokenizer state and rebuild embedding layer.

        Args:
            state: Tokenizer state dictionary

        Returns:
            self for method chaining
        """
        self.tokenizer.load_state(state)

        # Rebuild embedding layer with loaded vocabulary size
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self.tokenizer.pad_token
        )

        # Move to device
        self.embedding = self.embedding.to(self._device)

        logger.info(f"Tokenizer state loaded, embedding layer rebuilt:")
        logger.info(f"  Vocabulary size: {self.tokenizer.vocab_size}")

        return self

    def init_embeddings_from_bert(
        self,
        bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze: bool = False
    ) -> 'CNNFeatureExtractor':
        """
        Initialize word embeddings from a BERT model.

        For each word in our vocabulary:
        1. Tokenize with BERT tokenizer → get subword tokens
        2. Look up subword embeddings from BERT
        3. Average the subword embeddings
        4. Project to our embedding dimension

        Args:
            bert_model_name: HuggingFace model name for BERT
            freeze: If True, freeze embeddings after initialization

        Returns:
            self for method chaining
        """
        from transformers import AutoTokenizer, AutoModel

        if not self.tokenizer._is_fitted:
            raise RuntimeError(
                "Tokenizer must be fitted before initializing embeddings from BERT. "
                "Call fit_tokenizer() first."
            )

        logger.info(f"Initializing embeddings from {bert_model_name}")

        # Load BERT tokenizer and model
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_model = AutoModel.from_pretrained(bert_model_name)
        bert_model.eval()

        # Get BERT embedding weights
        bert_embeddings = bert_model.embeddings.word_embeddings.weight  # (vocab, 768)
        bert_dim = bert_embeddings.size(1)

        # Create projection layer if dimensions differ
        if bert_dim != self._embedding_dim:
            projection = nn.Linear(bert_dim, self._embedding_dim, bias=False)
            nn.init.xavier_uniform_(projection.weight)
            projection = projection.to(bert_embeddings.device)
        else:
            projection = None

        # Map each vocabulary word to BERT embedding
        initialized_count = 0
        with torch.no_grad():
            for word, idx in self.tokenizer.word_to_id.items():
                if word in [self.tokenizer.PAD_TOKEN, self.tokenizer.UNK_TOKEN]:
                    continue

                # Tokenize word with BERT tokenizer
                bert_tokens = bert_tokenizer.encode(word, add_special_tokens=False)

                if bert_tokens:
                    # Average subword embeddings
                    subword_embs = bert_embeddings[bert_tokens]  # (num_subwords, 768)
                    word_emb = subword_embs.mean(dim=0)  # (768,)

                    # Project to our embedding dimension
                    if projection is not None:
                        word_emb = projection(word_emb)

                    self.embedding.weight.data[idx] = word_emb.to(self._device)
                    initialized_count += 1

        # Freeze if requested
        if freeze:
            self.embedding.weight.requires_grad = False
            logger.info("  Embeddings frozen (requires_grad=False)")

        logger.info(f"  Initialized {initialized_count}/{len(self.tokenizer.word_to_id)} "
                    f"word embeddings from BERT")

        return self

    def init_filters(
        self,
        texts: List[str],
        explicit_concepts: Optional[Dict[str, List[str]]] = None,
        num_latent_per_kernel: int = 64,
        bert_model_name: Optional[str] = None,
        freeze: bool = False
    ) -> 'CNNFeatureExtractor':
        """
        Initialize CNN filters from explicit concepts and k-means clustering.

        For each kernel size k with num_filters=N:
        - First len(concepts[k]) filters: from explicit concepts
        - Next num_latent_per_kernel filters: from k-means on training n-grams
        - Remaining: keep random initialization

        Args:
            texts: Training texts for k-means clustering of n-grams
            explicit_concepts: Dict mapping kernel_size (as string) to list of concept phrases
            num_latent_per_kernel: Number of k-means derived filters per kernel size
            bert_model_name: If provided, use BERT embeddings for concepts
                            (otherwise use current embedding layer)
            freeze: If True, freeze all conv layer weights after initialization

        Returns:
            self for method chaining
        """
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans

        if not self.tokenizer._is_fitted:
            raise RuntimeError(
                "Tokenizer must be fitted before initializing filters. "
                "Call fit_tokenizer() first."
            )

        logger.info("Initializing CNN filters from concepts and k-means")

        # Parse explicit concepts
        if explicit_concepts is None:
            explicit_concepts = {}

        # Normalize keys to int
        concepts_by_kernel = {
            int(k): v for k, v in explicit_concepts.items()
        }

        # Helper to get word embedding
        def get_word_embedding(word: str) -> torch.Tensor:
            """Get embedding for a word from current embedding layer."""
            idx = self.tokenizer.word_to_id.get(
                word.lower(),
                self.tokenizer.UNK_ID
            )
            return self.embedding.weight[idx].detach().clone()

        for conv in self.convs:
            kernel_size = conv.kernel_size[0]
            num_filters = conv.out_channels

            concepts = concepts_by_kernel.get(kernel_size, [])
            num_explicit = len(concepts)
            filter_idx = 0

            # Phase 1: Initialize filters from explicit concepts
            if concepts:
                logger.info(f"  Kernel {kernel_size}: {num_explicit} explicit concept filters")
                with torch.no_grad():
                    for concept in concepts:
                        if filter_idx >= num_filters:
                            break

                        concept_words = concept.lower().split()[:kernel_size]

                        # Pad if concept has fewer words than kernel size
                        while len(concept_words) < kernel_size:
                            concept_words.append(concept_words[-1] if concept_words else "the")

                        # Build filter: stack word embeddings
                        filter_weights = torch.stack([
                            get_word_embedding(w) for w in concept_words
                        ], dim=1)  # (embed_dim, kernel_size)

                        conv.weight.data[filter_idx] = filter_weights.to(self._device)
                        filter_idx += 1

            # Phase 2: Initialize filters from k-means clustering of n-grams
            remaining_for_kmeans = min(
                num_latent_per_kernel,
                num_filters - filter_idx
            )

            if remaining_for_kmeans > 0 and texts:
                logger.info(f"  Kernel {kernel_size}: {remaining_for_kmeans} k-means filters")

                # Extract n-grams from training texts
                ngram_embeddings = []
                max_ngrams = 100000  # Limit for memory

                for text in texts:
                    tokens = self.tokenizer._tokenize(text)
                    for i in range(len(tokens) - kernel_size + 1):
                        ngram = tokens[i:i+kernel_size]

                        # Get embedding for each token
                        token_embs = []
                        for tok in ngram:
                            idx = self.tokenizer.word_to_id.get(tok, self.tokenizer.UNK_ID)
                            token_embs.append(
                                self.embedding.weight[idx].detach().cpu()
                            )

                        # Stack to (embed_dim, k) then flatten
                        ngram_emb = torch.stack(token_embs, dim=1)  # (embed_dim, k)
                        ngram_embeddings.append(ngram_emb.flatten().numpy())

                        if len(ngram_embeddings) >= max_ngrams:
                            break
                    if len(ngram_embeddings) >= max_ngrams:
                        break

                if len(ngram_embeddings) >= remaining_for_kmeans:
                    # Run k-means
                    ngram_array = np.stack(ngram_embeddings)
                    kmeans = MiniBatchKMeans(
                        n_clusters=remaining_for_kmeans,
                        random_state=42,
                        n_init=3,
                        batch_size=min(1000, len(ngram_embeddings))
                    )
                    kmeans.fit(ngram_array)

                    # Reshape centers to filter weights
                    centers = torch.tensor(
                        kmeans.cluster_centers_,
                        dtype=torch.float32
                    )
                    centers = centers.reshape(
                        remaining_for_kmeans,
                        self._embedding_dim,
                        kernel_size
                    )

                    with torch.no_grad():
                        for i in range(remaining_for_kmeans):
                            if filter_idx >= num_filters:
                                break
                            conv.weight.data[filter_idx] = centers[i].to(self._device)
                            filter_idx += 1
                else:
                    logger.warning(
                        f"  Kernel {kernel_size}: Not enough n-grams "
                        f"({len(ngram_embeddings)}) for {remaining_for_kmeans} k-means filters"
                    )

            # Remaining filters keep random initialization
            remaining = num_filters - filter_idx
            if remaining > 0:
                logger.info(f"  Kernel {kernel_size}: {remaining} random filters (unchanged)")

        # Freeze conv weights if requested
        if freeze:
            for conv in self.convs:
                conv.weight.requires_grad = False
                if conv.bias is not None:
                    conv.bias.requires_grad = False
            logger.info("  CNN filters frozen (requires_grad=False)")

        return self

    @property
    def output_dim(self) -> int:
        """Total output dimension after optional projection."""
        if self._projection_dim is not None:
            return self._projection_dim
        return self.hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts using CNN.

        Args:
            texts: List of text strings to encode

        Returns:
            Feature tensor: (batch, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Embed words: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)

        # Apply attention mask (zero out padding)
        x = x * attention_mask.unsqueeze(-1).float()

        # Transpose for conv1d: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply each conv layer and max pool
        conv_outputs = []
        for conv in self.convs:
            # Conv: (batch, embed_dim, seq_len) -> (batch, num_filters, seq_len)
            h = F.relu(conv(x))

            # Apply mask to conv output (zero out padding positions)
            h = h * attention_mask.unsqueeze(1).float()

            # Global max pool: (batch, num_filters, seq_len) -> (batch, num_filters)
            # Use masked max: set padding to large negative before max
            h_masked = h.masked_fill(~attention_mask.unsqueeze(1).bool(), float('-inf'))
            h_pooled = h_masked.max(dim=2)[0]

            # Handle all-padding case (shouldn't happen with valid texts)
            h_pooled = torch.where(
                torch.isinf(h_pooled),
                torch.zeros_like(h_pooled),
                h_pooled
            )

            conv_outputs.append(h_pooled)

        # Concatenate all conv outputs: (batch, num_filters * num_kernels)
        features = torch.cat(conv_outputs, dim=1)

        # Apply dropout
        features = self.dropout(features)

        # Apply projection if configured
        if self.projection is not None:
            features = self.projection(features)

        # Apply final normalization
        #output = self.feature_norm(features)
        output=features
        
        return output

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
