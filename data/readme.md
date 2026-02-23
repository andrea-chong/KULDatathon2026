# Data Description: User Fingerprint Matrix (`df_user_fp`)

This document describes the structure and logic of the `df_user_fp` dataset. This matrix serves as a multi-dimensional "Digital Fingerprint" for learners, integrating macro-level behavioral statistics, micro-level proficiency scores, and exposure intensity across specific linguistic lexemes.

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
```
