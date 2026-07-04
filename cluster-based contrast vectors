# Cluster-Based Contrast Vectors

The embedding-contrast path now includes an optional contrast-basis family built
from patient clusters. The goal is to avoid forcing all text signal into one
global mean-difference vector when several confounders or effect modifiers may
live in different regions of the patient embedding space.

The existing global contrasts are still computed first. Cluster-based contrast
vectors add extra retrieval directions; they do not replace the treatment,
outcome, confounder, cell-interaction, or R-style contrasts.

## Recipe

For each discovery fold:

1. Retained chunk embeddings are averaged into patient-level embeddings, then
   normalized and optionally residualized in the same way as the existing
   embedding-contrast path.
2. Patients with finite embeddings are clustered with MiniBatchKMeans.
3. Within each sufficiently large cluster, the generator computes local
   treatment contrasts:

   ```text
   d_s = mean_embedding(T=1, S=s) - mean_embedding(T=0, S=s)
   ```

4. Within clusters that have enough treatment/outcome cell support, it also
   computes local residualized treatment-outcome interaction contrasts:

   ```text
   r_s = residualize(
       mean(T=1,Y=1,S=s) - mean(T=1,Y=0,S=s)
       - mean(T=0,Y=1,S=s) + mean(T=0,Y=0,S=s),
       basis=[local treatment contrast, local outcome contrast]
   )
   ```

5. Each local vector is normalized and weighted by `sqrt(n_cluster)`.
6. The weighted local vectors are stacked into a matrix and decomposed with SVD.
   The top right singular vectors become cluster-based retrieval directions.

This is an uncentered PCA over local contrast vectors. Equivalently, it finds
the leading eigenvectors of:

```text
sum_s n_s * normalize(d_s) normalize(d_s)'
```

for local treatment contrasts, and the analogous matrix for local interaction
contrasts.

## Interpretation

A global contrast answers:

```text
What is the average embedding difference between groups?
```

A cluster-based contrast basis asks:

```text
What recurring kinds of group differences appear across patient subgroups?
```

The components are not arbitrary pieces of one vector. Each component summarizes
a family of real within-cluster contrasts. Evidence records include
`cluster_component_loadings`, which show which patient clusters contributed most
strongly to each component, their local sample counts, and local contrast norms.

## Configuration

The controls live under:

```text
architecture.multi_model_agentic_forest.embedding_contrast
```

Relevant fields:

```text
include_cluster_contrast_vectors: true
cluster_contrast_n_clusters: 10
cluster_contrast_max_components: 5
cluster_contrast_min_cluster_size: 24
cluster_contrast_min_group_size: 8
cluster_contrast_min_cell_size: 4
cluster_contrast_top_loadings: 5
cluster_contrast_random_state: 42
cluster_contrast_kmeans_n_init: 20
```

If too few patients or too few usable local contrasts are available, the evidence
payload records the skip reason in `cluster_contrast_vectors` and continues with
the standard global contrasts.

## Guardrails

Cluster-based vectors can surface additional axes, but they can also amplify
small-cell noise. The implementation therefore requires minimum cluster, group,
and treatment/outcome cell counts before a local contrast enters the SVD basis.
For stochastic binary outcomes, interaction components should be treated as
hypothesis-generation evidence and judged by cross-fold stability, permutation
checks, and whether retrieved chunks suggest plausible pre-treatment variables.
