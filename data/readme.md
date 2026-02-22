# Data Description: User Fingerprint Matrix (`df_user_fp`)

This document describes the structure and logic of the `df_user_fp` dataset, which serves as a "Digital Fingerprint" for learners. It integrates macro-level behavioral statistics with micro-level proficiency scores across specific linguistic lexemes.

## 1. Behavioral Features (User-Level Statistics)

These features capture the general learning habits and the "age" of a user's account in terms of exposure and intensity.

| Column Name            | Definition                                  | Logic & Significance                                                                                                                           |
| :--------------------- | :------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| **`user_id`**          | Unique Identifier                           | The primary key used to link sessions and historical performance.                                                                              |
| **`max_history_seen`** | $\max(history\_seen)$                       | **Experience Index**: Reflects the maximum cumulative practice sessions a user has reached for any word.                                       |
| **`vocab_size`**       | `count(distinct lexeme_id)`                 | **Breadth Index**: Measures the variety of the user's vocabulary within the target language.                                                   |
| **`learning_speed`**   | $\ln(1 + \frac{vocab\_size}{days\_active})$ | **Efficiency Metric**: Calculates new words acquired per day. Log-transformed to handle the long-tail distribution of high-intensity learners. |

---

## 2. Lexeme Mastery Features (Knowledge Fingerprint)

This high-dimensional section (from `lexeme_0` to `lexeme_n`) represents the user's proficiency boundary based on specific vocabulary items.

- **Column Format**: `lexeme_{lexeme_code}`
- **Value Definition**: `user_ability_index`
  - **Formula**: $Accuracy^{Recall\_Rate}$
  - **Mathematical Logic**:
    - **Non-linear Weighting**: This index prioritizes performance on difficult words. A decent accuracy on a low-recall (difficult) word yields a score near 1.0, while accuracy on high-recall (easy) words remains proportionate.
    - **Proficiency Boundary**: It distinguishes "advanced users" who handle complex grammar from "beginners" who master only high-frequency vocabulary.
- **Missing Values**: `NaN` values are filled with `0`, indicating the user has not yet encountered or practiced that specific lexeme.

---

## 3. Data Structure & Technical Specs

- **Shape**: `[Total Users] x [4 Behavioral + N Lexeme Columns]`
- **Sparsity**: Extremely High. Most users interact with only a fraction of the total available lexemes.
- **Normalization Recommendation**:
  - Apply `StandardScaler` to behavioral features before clustering.
  - Apply `Dimensionality Reduction` (e.g., **TruncatedSVD** or **PCA**) to the lexeme columns to condense the 12,000+ dimensions into latent "ability vectors."

---

## 4. Analytical Objectives

1.  **User Clustering**: Segment learners into personas (e.g., "The Perfectionist," "The Speed Learner," or "The High-Difficulty Explorer").
2.  **Churn Prediction**: Use the knowledge fingerprint to detect stagnation or "plateaus" in learning mastery.
3.  **Similarity Analysis**: Calculate **Cosine Similarity** between user fingerprints to build a Collaborative Filtering system for personalized content recommendations.
