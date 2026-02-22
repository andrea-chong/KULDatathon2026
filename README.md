# Buddy System for Language Learning

# Team Team - KU Leuven Datathon 2026

## Objective

Goal: Identify the most suitable learning buddy for each user.
Who shall be my learning buddy?
criteria: learners of the same language (doesn’t matter if ui_language) is different
focus on learning_language==’pt’ dataset
quantify distance between users (based on history of lexeme correctness)

---

### Features for clustering

1. Lexeme Performance
   • Historical correctness per lexeme
   • Weighted by lexeme difficulty
   • Captures mastery rather than raw accuracy

2. Lexeme Exposure
   • Historical exposure ("history_seen")
   • Standardised across users
   • Reflects familiarity with vocabulary

3. Learning Progress
   • User level proxy:
   • Maximum "history_seen" per user
   • Can be up-weighted to reflect experience

4. Learning Speed

\text{learning*speed} = \frac{\text{vocab_size}}{\text{timestamp}*{max} - \text{timestamp}\_{min}}
Measures how quickly a learner acquires new vocabulary.

### Clustering algorithms

1. K-Means (Baseline)
   • Simple and interpretable
   • Provides a benchmark for other methods

2. DBSCAN (Density-Based)
   • Detects dense clusters and labels outliers as noise
   • Useful for filtering bots or extreme users
   • Best when learners form natural “skill pockets”

3. Agglomerative Hierarchical Clustering
   • Builds a hierarchy (tree) of learners
   • Enables analysis at multiple granularity levels
   • Helpful for visualising pairwise learner distances

4. Autoencoder Embeddings
   • Learns compact latent representations
   • Captures nonlinear relationships in learning behaviour
   • Embeddings can be clustered with standard algorithms

### Distance Metrics

- Euclidean DistanceStandard "as the crow flies" distance. Best for normalized scores.
- Cosine SimilarityFocuses on the orientation of the features. Good if you care more about the ratio of lessons-to-accuracy than the raw count.
- Manhattan DistanceBetter if you have high-dimensional data or want to be less sensitive to extreme outliers (like a user who took 1,000 lessons but has 0% accuracy).

---

### Tasks

Create feature matrix - by Mon 23rd Feb
add Dataset B features
Clustering tasks - by Wed 25th Feb before we meet
K-means
DBSCAN
Agglo hierarchical
Auto-encoder - Embeddings - by Tues 24th Feb
Powerpoint - outline before we meet on Wed 25th Feb

This is the team repo for the KUL Datathon 2026 :)
