# Data Description: User Fingerprint Matrix

This document describes the structure and logic of the `/raw/users_fingerprint.csv` dataset. This matrix serves as a multi-dimensional "Digital Fingerprint" for learners, integrating macro-level behavioral statistics, micro-level proficiency scores, and exposure intensity across specific linguistic lexemes.

---

## Overall Data Pipeline & Matchmaking Logic

Our engine transforms raw learning logs into high-dimensional user "fingerprints" through a multi-stage pipeline:

1.  **Lexeme Semantic Embedding**:
    Generating vector representations for all vocabulary. We use **HDBSCAN** on these semantic embeddings (found in `/final/lexeme_embed_cluster_results.parquet`) to identify **81 stable lexeme archetypes**.
2.  **Feature Fusion**:
    Integrating **User Historical Performance** with these **81 Semantic Archetypes**. This captures not just _if_ a user knows a word, but their mastery across semantically related linguistic groups.
3.  **User Clustering**:
    Utilizing **HDBSCAN** on `/final/user_fingerprint_B_lex_clusters_scaled.csv` to identify stable **Learner Archetypes** (The Twins & The Opposites).

---

## 1. Behavioral Features (User-Level Statistics)

These features capture general learning habits and the overall progression of a user's account.

| Column Name            | Definition                                  | Logic & Significance                                                                                                                    |
| :--------------------- | :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------- |
| **`user_id`**          | Unique Identifier                           | The primary key used to link sessions and historical performance.                                                                       |
| **`max_history_seen`** | $\max(history\_seen)$                       | **Experience Index**: Reflects the maximum cumulative practice sessions a user has reached for any single word.                         |
| **`vocab_size`**       | `count(distinct lexeme_id)`                 | **Breadth Index**: Measures the total variety of vocabulary items the user has encountered.                                             |
| **`learning_speed`**   | $\ln(1 + \frac{vocab\_size}{days\_active})$ | **Efficiency Metric**: Calculates new words acquired per day. Log-transformed to normalize the distribution of high-intensity learners. |

---

## 2. Knowledge & Exposure Fingerprint (High-Dimensional)

The fingerprint is composed of two parallel sets of columns for each lexeme (where `n` is the `lexeme_code`).

### A. Proficiency Scores (`lexeme_{n}`)

- **Value**: `user_ability_index`
- **Formula**: $history_acc_rate^{lexeme_avg_recall}$
- **Significance**: Represents the user's mastery boundary. It uses non-linear weighting to reward users who maintain high accuracy on difficult (low-recall) words.

### B. Exposure Intensity (`lexeme_{n}_seen`)

- **Value**: $\max(history\_seen)$ for that specific lexeme.
- **Significance**: Represents the "Practice Depth." It tracks how many times a user has been exposed to a specific word.
- **Analytic Value**: Comparing `lexeme_{n}` with `lexeme_{n}_seen` allows the identification of learning efficiency (e.g., high mastery with low exposure vs. low mastery despite high exposure).

**Missing Values**: All `NaN` values in these columns are filled with `0`, indicating no interaction with that lexeme.

---

## 3. Data Structure & Technical Specs

- **Total Columns**: $4 + (2 \times N)$ (4 Behavioral + N Mastery + N Exposure columns).
- **Sparsity**: Extremely High. Most users only interact with a small subset of the total available lexemes.
- **Feature Engineering Note**:
  - Mastery features (`lexeme_{n}`) are focused on **quality** of learning.
  - Exposure features (`lexeme_{n}_seen`) are focused on **quantity** of effort.

---

## 4. Analytical Objectives

1.  **Learner Segmentation**: Use clustering to identify personas such as "The Natural" (High mastery, low exposure) vs. "The Grinder" (High mastery, high exposure).
2.  **Difficulty Bottleneck Analysis**: Identify specific lexemes where `history_seen` is high but `ability_index` remains low across the user base.
3.  **Dimensionality Reduction**: Given the doubled column count, using **TruncatedSVD** is highly recommended to extract latent features before feeding into any clustering or classification models.

---

## 5. Use information from Dataset B

Dataset B (prompt–translation corpus) is used to estimate how “useful / representative” each lexeme is in real prompt contexts, and convert that into a **lexeme weight**. We then use this weight to build an additional user × lexeme feature matrix.

### 5.1 Join Dataset B statistics onto each lexeme

1. Parse `word` from `lexeme_string` (keep the surface/lemma part before tags like `<...>`).
2. Convert `word` to a lookup token (`query_token`, usually the lemma after `/`, lowercased).
3. From Dataset B, compute per token:
   - `prompt_count`: number of prompts where the token appears
   - `prompt_coverage = prompt_count / total_prompts`
   - `frequency`: average probability mass of the token within prompts (based on translation probabilities)

Lexemes that never appear in Dataset B will have missing `frequency` / `prompt_coverage`.

### 5.2 Compute lexeme weight

For each lexeme \(l\), compute a raw weight:

```math
\mathrm{weight\_raw}(l)
=
\left(4p(1-p)\right)^{\beta}
\cdot
\left(\sqrt{f}\right)^{\delta}
\cdot
\left(\sqrt{\mathrm{cov}}\right)^{\gamma}
```

- \(p\): `global_correctness` from Dataset A
- \(f\): `frequency` from Dataset B (NaN → default `0.01`)
- `cov`: `prompt_coverage` from Dataset B (NaN → default `0.01`)

- Hyperparameters:

```math
\beta=1.0,\ \delta=0.5,\ \gamma=0.5
```

Then normalize:

```math
\mathrm{weight}(l) = \frac{\mathrm{weight\_raw}(l)}{\max(\mathrm{weight\_raw})}
```

So `weight ∈ [0, 1]`.

### 5.3 Build B-based user lexeme features

For each `(user_id, lexeme_code)` in Dataset A (Portuguese subset), compute:

```math
\mathrm{feature\_score}(u,l) = r(u,l)\cdot \sqrt{\mathrm{weight}(l)}
```

- \(r(u,l)\): `history_acc_rate` (user’s historical correctness on that lexeme)
- If a user has no record for a lexeme, the feature is `0`.

Pivot to a wide matrix (`lexeme_0 ... lexeme_2814`) and merge with the existing user fingerprint table to produce:

- `user_fingerprint_B.csv`
- `user_fingerprint_B_scaled.csv` (StandardScaler applied to all numeric columns)

---

## 6. User Embedding Generation (SVD / SVD→AE)

We provide two pipelines to generate user-level embeddings from `df_user_fp` (user aggregated features), which contains:

- **Lexeme ability features**: `lexeme_*`
- **Lexeme exposure features**: `lexeme_*_seen`
- **Behavioral features**: `max_history_seen`, `vocab_size`, `learning_speed`

Both pipelines end with **L2 normalization** so the embeddings work well with cosine similarity / clustering.

We use the 2 piplines on both the direct user features and the features with information from dataset B (users_fingerprint_norm.csv and user_fingerprint_B_scaled.csv), resulting in 4 different datasets.

### 6.1 SVD Reduction + Behavioral Late Fusion

1. **Column grouping**
   - `ability_cols`: all `lexeme_*` columns excluding `*_seen`, `user_id`, and behavioral columns
   - `seen_cols`: all `lexeme_*_seen` columns
   - `beh_cols`: `max_history_seen`, `vocab_size`, `learning_speed`

2. **Behavioral preprocessing**
   - Fill missing values with 0
   - Clip each feature to \([-5, 5]\)
   - Down-weight with `w_beh = 0.5`

3. **Dimensionality reduction (linear)**
   - Ability block: `TruncatedSVD(n_components=64)` → `E_ability` (64-d)
   - Seen block: `log1p` + `TruncatedSVD(n_components=32)` → `E_seen` (32-d), then down-weight with `w_seen = 0.5`

4. **Concatenation & output**
   - Concatenate `[E_ability, E_seen, B]` → **99-d** embedding
   - L2 normalize per user
   - Export: `user_embedding_svd_ability_seen_beh.csv`

### 6.2 SVD(512) → AutoEncoder(128) + Behavioral Late Fusion

1. **Build a 512-d SVD representation**
   - Ability: `TruncatedSVD(n_components=384)` → `Z_a`
   - Seen: `log1p` + `TruncatedSVD(n_components=128)` → `Z_s`, then down-weight with `W_SEEN = 0.5`
   - Concatenate `[Z_a, W_SEEN * Z_s]` → `Z` (512-d)
   - Standardize `Z` with `StandardScaler` to stabilize AE training

2. **Train a small AutoEncoder (non-linear compression)**
   - Architecture: `512 → 256 → 128 → 256 → 512` (LayerNorm + GELU + Dropout)
   - Objective: reconstruct `Z` using MSE loss
   - Train/val split: 80/20, with early stopping (best validation loss)
   - Output embedding: encoder bottleneck → **128-d** (`emb_ae`)

3. **Behavioral late fusion & output**
   - Behavioral features are **not used to train the AE**
   - Fill missing values with 0, optionally standardize, clip to \([-5, 5]\), and down-weight with `W_BEH = 0.5`
   - Concatenate `[emb_ae, B]` → **131-d** embedding
   - L2 normalize per user
   - Export: `user_embedding_svd512_ae128_plus_beh.csv`

## 7. Lexical-level Embeddings and clustering

We use semantic and morphological features of lexemes to cluster them into groups as an alternative method to reduce dimensionality of lexical features (`lexeme_*` and `lexeme_*_seen`)

1. Get the top (max 10) most frequently used translated sentences in which each lexeme is used.
2. Use [CALE-XLLEX](https://huggingface.co/gabrielloiseau/CALE-XLLEX) model to generate semantic embeddings for the target lexeme in context.
3. Concatenate embeddings with morphological features of lexeme
4. Reduce dimensionality with UMAP
5. Run HDBSCAN clustering to assign each lexeme to a cluster (~402 not assigned to any cluster)
