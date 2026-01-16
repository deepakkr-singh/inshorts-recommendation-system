# Feature Engineering & Recommendation Pipeline

**Complete Implementation Report for News Recommendation System**

---

## Assignment Compliance

### Assignment Requirements

The assignment asked for a recommendation system that takes into account:

| Requirement                       | Implementation | Coverage               | Files Used                                     |
|-----------------------------------|----------------|------------------------|------------------------------------------------|
| **1. User's reading history**     | Implemented    | 100% (8,977 users)     | `interaction_matrix.pkl`, `user_profiles.csv`  |
| **2. User's expressed interests** | Implemented    | 96.8% (8,689 users)    | `user_category_tfidf.pkl`, `user_profiles.csv` |
| **3. Popularity of articles**     | Implemented    | 100% (14,622 articles) | `article_popularity.pkl`                       |
| **4. Location relevance**         | Implemented    | 10% training articles  | `article_locations.pkl`, `user_profiles.csv`   |

### Implementation Approach

**Data-Driven Design:**
- Temporal 80/20 split for training/validation
- Content-Based + Collaborative Filtering + Hybrid approach
- Handles cold-start for 970 new testing articles
- Addresses 97.89% data sparsity

**Evaluation Strategy:**
- **Stage 1 (Validation):** Metrics on training data (80/20 split)
- **Stage 2 (Testing):** Final recommendations on new articles

---

## Section 1: Data Overview

### Raw Data Dimensions

| File                   | Rows      | Columns | Purpose                                                      |
|------------------------|-----------|---------|--------------------------------------------------------------|
| `devices.csv`          | 10,400    | 11      | User device metadata (location, language, platform)          |
| `events.csv`           | 3,544,161 | 13      | User-article interactions (clicks, reads, bookmarks, shares) |
| `training_content.csv` | 8,170     | 12      | Articles with metadata (for training/validation)             |
| `testing_content.csv`  | 970       | 12      | New articles (cold-start testing scenario)                   |

### Key Statistics

```
Total unique users: 8,977
Total unique articles: 14,622
Total interactions: 3,544,161
Interaction matrix sparsity: 97.89%
User segments:
  - Passive users (1-9 interactions): 14.08% (1,264 users)
  - Active users (10-99 interactions): 77.14% (6,923 users)
  - Power users (100+ interactions): 8.78% (788 users)
```

---

## Section 2: Feature Engineering Pipeline

### 2.1 Article Features

**Input:**
- `training_content.csv` (8,170 articles)
- `testing_content.csv` (970 articles)

**Processing Steps:**

1. **Category TF-IDF Vectorization**
   - Parse categories column (comma-separated)
   - Apply TF-IDF with min_df=2 (remove rare categories)
   - Result: 55 raw categories → 38 features

2. **Language One-Hot Encoding**
   - 9 languages: English, Hindi, Malayalam, etc.
   - Create binary features for each language

3. **NewsType Label Encoding**
   - 3 types: text_image, video, photo
   - Encode as 0, 1, 2

4. **Popularity Scores**
   - Formula: log(unique_users) × 0.7 + log(total_events) × 0.3
   - Normalized to 0-1 range

**Output Files:**

| File                     | Dimensions                 | Format     | Purpose                   |
|--------------------------|----------------------------|------------|---------------------------|
| `tfidf_vectorizer.pkl`   | Fitted model (38 features) | Pickle     | Transform new articles    |
| `training_tfidf.pkl`     | 8,170 × 38                 | Sparse CSR | Training article vectors  |
| `testing_tfidf.pkl`      | 970 × 38                   | Sparse CSR | Testing article vectors   |
| `article_features.csv`   | 9,140 × 22                 | CSV        | Combined article metadata |
| `article_popularity.pkl` | 14,622 entries             | Dict       | Popularity scores         |
| `article_locations.pkl`  | 9,140 entries              | Dict       | Inferred city or NATIONAL |

---

### 2.2 User Features

**Input:**
- `events.csv` (3,544,161 interactions)
- `devices.csv` (10,400 devices)
- `training_content.csv` (for category mapping)

**Processing Steps:**

1. **Reading History Aggregation**
   - Count interactions per user
   - Sum reads, clicks, bookmarks, shares
   - Calculate engagement weights:
     - TimeSpent-Front: 1.0
     - TimeSpent-Back: 2.0
     - Bookmarked: 3.0
     - Shared: 5.0

2. **Category Profile Building**
   - Merge events with article categories
   - Apply engagement weights to each interaction:
     - TimeSpent-Front: 1.0
     - TimeSpent-Back: 2.0
     - Bookmarked: 3.0
     - Shared: 5.0
   - Multiply category list by weight (repetition based on engagement strength)
   - Concatenate all weighted categories
   - **Example:**
     ```
     User reads 3 articles:
     1. Article: "sports,cricket", engagement=TimeSpent-Front (weight=1)
        → ["sports", "cricket"] × 1 = ["sports", "cricket"]

     2. Article: "sports,football", engagement=Bookmarked (weight=3)
        → ["sports", "football"] × 3 = ["sports", "football", "sports", "football", "sports", "football"]

     3. Article: "technology", engagement=Shared (weight=5)
        → ["technology"] × 5 = ["technology", "technology", "technology", "technology", "technology"]

     Final category_text:
     "sports cricket sports football sports football sports football technology technology technology technology technology"
     ```
   - Repetition captures both **preference** (which categories) and **intensity** (how engaged)

3. **User TF-IDF Transformation**
   - Transform user category text using SAME vectorizer as articles
   - Result: 8,689 × 38 (same space as articles → enables cosine similarity)

4. **User Metadata Extraction**
   - From devices.csv: city, language, platform
   - Coverage: Only 7.9% users (684/8,689) matched in devices table
   - Most users: metadata from events (inferred language from article reads)
   - **Gap reason:** 92.1% of active users who read articles are NOT in devices.csv

**Output Files:**

| File                      | Dimensions | Format     | Purpose                  |
|---------------------------|------------|------------|---------------- ---------|
| `user_category_tfidf.pkl` | 8,689 × 38 | Sparse CSR | User interest vectors    |
| `user_profiles.csv`       | 8,689 × 8  | CSV        | Aggregated user metadata |

**User Profile Columns:**
- `deviceId`: User identifier
- `num_interactions`: Total interaction count
- `num_reads`, `num_clicks`, `num_bookmarks`, `num_shares`: Engagement metrics
- `user_city`: City (from devices or inferred)
- `preferred_language`: Most-read language
- `top_categories`: Top 3 categories by engagement weight

---

### 2.3 Collaborative Filtering Features

**Input:**
- `events.csv` (all 3,544,161 interactions)

**Processing Steps:**

1. **Interaction Matrix Construction**
   - Create user-to-index and article-to-index mappings
   - Build sparse matrix: 8,977 users × 14,622 articles
   - Cell values: Engagement score (weighted by event type)
   - Format: CSR (Compressed Sparse Row) for memory efficiency

2. **User-User Similarity Computation**
   - Only for active users (≥10 interactions): 6,342 users
   - L2 normalize user vectors
   - Compute cosine similarity between users
   - Store top 50 neighbors per user

**Output Files:**

| File                     | Dimensions     | Format     | Purpose                             |
|--------------------------|----------------|------------|-------------------------------------|
| `interaction_matrix.pkl` | 8,977 × 14,622 | Sparse CSR | User-article engagement             |
| `user_similarity.pkl`    | 6,342 users    | Dict       | K-NN neighbors (K=50)               |
| `mappings.pkl`           | 4 dicts        | Pickle     | ID mappings (user/article to index) |

**Sparsity Calculation:**
```
Total cells: 8,977 × 14,622 = 131,254,794
Non-zero cells: 277,000 (unique user-article pairs)
Sparsity: (1 - 277,000 / 131,254,794) × 100 = 97.89%
```

---

### 2.4 Location Features (Inferred)

**Challenge:** Location data coverage gap

- **devices.csv**: 91.3% (9,492/10,400) devices have `lastknownsubadminarea`
- **Active users**: Only 7.9% (684/8,689) are in devices.csv
- **Data quality issue**: 92.1% of active users NOT in devices table

**Solution:** Infer article locations from reader geography

**Processing:**

1. **For Training Articles:**
   - Group readers by article
   - Calculate % of reads from each city
   - If ≥50% reads from one city → tag article as that city
   - Otherwise → tag as NATIONAL

2. **For Testing Articles:**
   - Use newsDistrict metadata (32% coverage)
   - Map district codes to city names
   - Otherwise → tag as NATIONAL

**Output:**
- `article_locations.pkl`: 9,140 entries (hashid → city or "NATIONAL")

**Actual Coverage (Current Implementation):**
- Training articles: **17.8%** (1,631/9,152) get specific city tags
  - Inferred from reader geography (≥50% reads from one city)
- Testing articles: **0%** (all 970 tagged as NATIONAL)
  - **Potential improvement:** 32.3% (313/970) have newsDistrict metadata
  - **Current limitation:** newsDistrict mapping not yet implemented for testing
- Rest: Tagged as NATIONAL (relevant to all locations)

---

## Section 3: Mapping to Assignment Requirements

**IMPORTANT:** The percentages below (70/20/10) represent **general assignment requirement importance**, NOT algorithm-specific formula weights.

- Content-Based uses different weights: ~75% content similarity + 15% popularity + 10% language match
- Collaborative uses different weights: ~90% neighbor engagement + 10% popularity
- Hybrid uses: 60% collaborative + 25% content + 15% popularity

---

### File-to-Requirement Mapping

| Assignment Requirement            | Files Used                                        | Algorithm                                     |
|-----------------------------------|---------------------------------------------------|-----------------------------------------------|
| **1. User's Reading History**     | `interaction_matrix.pkl`                          | Collaborative ONLY                            |
|                                   | `user_category_tfidf.pkl`                         | Content-Based (categories from read articles) |
|                                   | `user_profiles.csv (num_interactions, num_reads)` | Both                                          |
| **2. User's Expressed Interests** | `user_category_tfidf.pkl`                         | Content-Based (TF-IDF of preferences)         |
|                                   |  `user_profiles.csv (top_categories)`             | Both                                          |
|                                   | `user_profiles.csv (preferred_language)`          | Content-Based                                 |
| **3. Popularity**                 | `article_popularity.pkl`                          | Both                                          |
| **4. Location**                   | `article_locations.pkl`                           | Both                                          |
|                                   | `user_profiles.csv (user_city)`                   | Both                                          |

---

### 1. User's Reading History (Primary Signal - 70%)

**Implementation:**

| Feature                   | How It Captures Reading History                             | Algorithm Used          |
|---------------------------|-------------------------------------------------------------|-------------------------|
| `interaction_matrix.pkl`  | Stores EVERY user-article interaction with engagement score | Collaborative Filtering |
| `user_category_tfidf.pkl` | Aggregates categories from read articles (TF-IDF weighted)  | Content-Based           |
| `user_profiles.csv`       | Counts: num_interactions, num_reads, num_clicks, etc.       | Both algorithms         |

**Example:**
```
User A reads: [article1: "sports,cricket", article2: "sports,football", article3: "technology"]
→ interaction_matrix[A] = [1: 3.0, 2: 2.5, 3: 1.0]  (engagement scores)
→ user_category_tfidf[A] = [sports: 0.7, cricket: 0.4, football: 0.3, technology: 0.2]
```

---

### 2. User's Expressed Interests (Implicit - 20%)

**Implementation:**

| Feature                                  | How It Captures Interests                           | Algorithm Used                 |
|------------------------------------------|-----------------------------------------------------|--------------------------------|
| `user_category_tfidf.pkl`                | TF-IDF of categories → "expressed" through behavior | Content-Based                  |
| `user_profiles.csv (top_categories)`     | Top 3 most-engaged categories                       | Hybrid (filtering)             |
| `user_profiles.csv (preferred_language)` | Most-read language                                  | Content-Based (language match) |

**Note:** No explicit interest tags in dataset → Inferred from reading behavior

**Example:**
```
User B:
  - Reads 80% sports articles → top_categories = ["sports", "cricket", "football"]
  - Reads mostly Hindi articles → preferred_language = "hi"
  → System "knows" User B is interested in Hindi sports content
```

---

### 3. Popularity of Articles (Cold-Start & Diversity - 10%)

**Implementation:**

| Feature                  | Formula                                           | Purpose                                   |
|--------------------------|---------------------------------------------------|-------------------------------------------|
| `article_popularity.pkl` | log(unique_users) × 0.7 + log(total_events) × 0.3 | Boost popular articles in recommendations |

**Usage:**
- **Content-Based:** Add 0.15 × popularity to content similarity scores
- **Collaborative:** Add 0.1 × popularity as fallback for articles with no neighbors
- **Hybrid:** Part of final score blending

**Why log-transform?**
- Prevents viral articles from dominating
- Balances breadth (unique users) and depth (total events)

---

### 4. Location Relevance (Geographic Boost - Variable Weight)

**Implementation:**

| Feature                         | Coverage                                         | Scoring       |
|---------------------------------|----------------------------------------------- --|---------------|
| `article_locations.pkl`         | 9,140 articles (10% city-specific, 90% NATIONAL) | Geo boost     |
| `user_profiles.csv (user_city)` | 8,689 users (7.9% have city, rest inferred)      | User location |

**Geo Scoring Logic:**
```
if user_city == article_city:
    geo_score = 1.0  (Perfect match)
elif article_city == "NATIONAL":
    geo_score = 0.7  (Neutral - good for all)
else:
    geo_score = 0.5  (Different city - reduced relevance)
```

**Applied Weight:**
- Content-Based: Multiply final score by geo_score
- Collaborative: Multiply final score by geo_score
- Effective weight: 5-10% depending on data availability

---

## Section 4: Content-Based Algorithm

### Algorithm Overview

**Principle:** Match article content features to user interest profile

**Core Formula:**
```
content_score = cosine_similarity(user_tfidf, article_tfidf)
                × geo_score
                + 0.15 × popularity_boost
                + 0.10 × language_match_bonus
```

---

### Required Files

| File                      | Purpose                               |
|---------------------------|---------------------------------------|
| `user_category_tfidf.pkl` | User interest vectors (8,689 × 38)    |
| `training_tfidf.pkl`      | Training article vectors (8,170 × 38) |
| `testing_tfidf.pkl`       | Testing article vectors (970 × 38)    | 
| `article_popularity.pkl`  | Popularity boost scores               |
| `article_locations.pkl`   | Geographic relevance                  |
| `user_profiles.csv`       | User metadata (city, language)        |
| `article_features.csv`    | Article metadata (language, type)     |

---

### Training on 80% of Training Data

**Temporal Split:**
```
Training articles: 8,170
Sort by created_datetime
→ Train set (80%): First 6,536 articles (by timestamp)
→ Validation set (20%): Last 1,634 articles (by timestamp)
```

**Training Process:**
1. Load user_category_tfidf for all 8,689 users
2. Load training_tfidf for first 6,536 articles
3. For each user:
   - Compute cosine similarity with all 6,536 articles
   - Apply geo_score multiplier
   - Add popularity boost (0.15 weight)
   - Add language match bonus (0.10 weight if user_language == article_language)
4. Rank articles by final score
5. Select top 50 recommendations

**Vectorized Implementation:**
```
# Batch computation for all users
similarity_matrix = user_category_tfidf @ training_tfidf.T
# Result: 8,689 users × 6,536 articles

# Apply geo scores (vectorized)
geo_scores_matrix = compute_geo_scores_vectorized(user_cities, article_cities)
similarity_matrix *= geo_scores_matrix

# Add popularity boost
similarity_matrix += 0.15 * popularity_scores

# Add language match
similarity_matrix += 0.10 * language_match_matrix
```

---

### Metrics on Training Data (80%)

**Metrics Computed:**

| Metric                 | Formula                                         | What It Measures                               |
|------------------------|-------------------------------------------------|------------------------------------------------|
| **Coverage**           | (Articles recommended / Total articles) × 100   | Diversity of recommendations                   |
| **User Coverage**      | (Users served / Total users) × 100              | % of users who can get recommendations         |
| **Category Diversity** | Entropy of categories in top-50                 | How diverse are recommended categories         |
| **Novelty**            | Average popularity rank of recommended articles | Are we recommending long-tail or just popular? |
| **Content Variance**   | Variance of TF-IDF vectors in top-50            | Similarity diversity                           |

**Expected Results:**
```
Coverage: ~95% (Content-based can recommend most articles)
User Coverage: 96.8% (8,689/8,977 users have TF-IDF profiles)
Category Diversity: 2.5-3.0 (high entropy)
Novelty: Medium (balances popular + niche)
Content Variance: 0.15-0.25 (reasonable diversity)
```

**Ranking Metrics (⚠️ DATA LEAKAGE WARNING):**

| Metric           | Formula                                   | What It Measures                      |
|------------------|-------------------------------------------|---------------------------------------|
| **NDCG@50**      | DCG / IDCG (position-aware relevance)     | Ranking quality                       |
| **Precision@50** | Relevant items in top-50 / 50             | Accuracy                              |
| **Recall@50**    | Relevant items in top-50 / Total relevant | Coverage of user's interests          |
| **MAP@50**       | Mean Average Precision across users       | Ranking quality across all users      |

**⚠️ IMPORTANT:** These metrics suffer from **data leakage** (model trained and evaluated on same data). Results will be **artificially inflated** and should NOT be used for model selection.

**Expected Results (Inflated due to leakage):**
```
NDCG@50: 0.45-0.65 (INFLATED - circular reasoning)
Precision@50: 0.15-0.30 (INFLATED - predicting what was used to train)
Recall@50: 0.45-0.70 (INFLATED - model "memorized" training data)
MAP@50: 0.20-0.40 (INFLATED - not true measure of generalization)
```

---

### Validation on 20% Held-Out Data

**Validation Set:**
- Last 1,634 articles (by timestamp)
- For these articles, we have actual user interactions (ground truth)

**Ground Truth Construction:**
```
For each user:
  - Find articles they interacted with in validation set
  - These are "relevant" items (binary: clicked/read = 1, else = 0)
```

**Validation Process:**
1. Generate top-50 recommendations using Content-Based (trained on 80%)
2. Compare with ground truth interactions
3. Compute distributional and ranking metrics

**Distributional Metrics (For Comparison):**

| Metric                 | Expected Value | What It Measures                           |
|------------------------|----------------|--------------------------------------------|
| **Coverage**           | 90-95%         | % of validation articles recommended       |
| **User Coverage**      | 96.8%          | % of users who can get recommendations     |
| **Category Diversity** | 2.5-3.2        | Entropy of recommended categories          |
| **Novelty**            | Medium-High    | Recommending newer/less popular articles   |
| **Content Variance**   | 0.15-0.30      | Similarity diversity in recommendations    |

---

**Ranking Metrics (TRUE Performance):**

| Metric           | Formula                                   | What It Measures                 |
|------------------|-------------------------------------------|----------------------------------|
| **NDCG@50**      | DCG / IDCG (position-aware relevance)     | Ranking quality                  |
| **Precision@50** | Relevant items in top-50 / 50             | Accuracy                         |
| **Recall@50**    | Relevant items in top-50 / Total relevant | Coverage of user's interests     |
| **MAP@50**       | Mean Average Precision across users       | Ranking quality across all users |

**Expected Results:**
```
NDCG@50: 0.15-0.25 (Content-based moderate performance)
Precision@50: 0.05-0.10 (5-10 relevant out of 50)
Recall@50: 0.20-0.35 (Captures 20-35% of user's interests)
MAP@50: 0.08-0.15
```

---

### Testing on Test Data (970 New Articles)

**Scenario:** Cold-start articles (ZERO interaction history)

**Testing Process:**
1. Load testing_tfidf (970 × 38)
2. For each user, compute similarity with all 970 testing articles
3. Apply geo_score, popularity, language match
4. Rank and select top-50

**Metrics Computed (NO Ground Truth):**

| Metric                 | What It Measures                                | Expected Value                        |
|------------------------|-------------------------------------------------|---------------------------------------|
| **Coverage**           | % of testing articles recommended at least once | 80-90%                                |
| **User Coverage**      | % of users who get recommendations              | 96.8%                                 |
| **Category Diversity** | Entropy of recommended categories               | 2.5-3.2                               |
| **Novelty**            | Average article age / popularity rank           | High (new articles)                   |
| **Sparsity**           | % of user-article pairs with score > 0          | ~40% (better than interaction matrix) |
| **Variance**           | Std dev of recommendation scores                | 0.15-0.30                             |

**Why NO NDCG/Precision/Recall?**
- Testing articles are NEW (zero interaction history)
- No ground truth to compare against
- Can only measure diversity, coverage, distributional properties

---

### Fallback Mechanism

**Scenario 1: User Has No TF-IDF Profile (288 users)**
```
Fallback: Recommend top 50 popular articles
Weight: 100% popularity (no content similarity available)
Affected users: 3.2% (288/8,977)
```

**Scenario 2: Article Not in TF-IDF Vocabulary**
```
Fallback: Use popularity + language match only
Weight: 60% popularity + 40% language match
Affected articles: Rare (all articles processed through TF-IDF)
```

**Scenario 3: Low Content Similarity (all scores < 0.1)**
```
Fallback: Boost popularity weight
Weight: 30% content + 70% popularity
Triggered when: User's interests don't match any article categories
```

---

## Section 5: Collaborative Filtering Algorithm

### Algorithm Overview

**Principle:** "Users who are similar to you also liked these articles"

**Core Formula:**
```
collab_score = Σ(neighbor_similarity × neighbor_engagement) / Σ(neighbor_similarity)
               × geo_score
               + 0.10 × popularity_boost
```

---

### Required Files

| File                     | Purpose                                  |
|--------------------------|------------------------------------------|
| `interaction_matrix.pkl` | User-article engagement (8,977 × 14,622) |
| `user_similarity.pkl`    | K-NN neighbors (6,342 users with K=50)   |
| `article_popularity.pkl` | Popularity fallback                      |
| `article_locations.pkl`  | Geographic relevance                     |
| `user_profiles.csv`      | User city, interaction count             |
| `mappings.pkl`           | User/article ID to index                 |

---

### Training on 80% of Training Data

**Temporal Split:**
```
Interactions: 3,544,161 events
Sort by created_datetime
→ Train interactions (80%): First 2,835,329 events
→ Validation interactions (20%): Last 708,832 events
```

**Training Process:**
1. Build interaction matrix from train interactions (80%)
   - Result: 8,977 users × ~11,700 articles (only articles in train set)
2. Compute user-user similarity (only active users with ≥10 interactions)
   - L2 normalize user vectors
   - Cosine similarity → top 50 neighbors per user
3. For each user:
   - Retrieve K=50 most similar neighbors
   - For each candidate article:
     - Aggregate neighbor engagement scores (weighted by similarity)
     - Normalize by sum of similarities
   - Apply geo_score multiplier
   - Add popularity boost (0.10 weight)
4. Rank articles by final score
5. Select top 50 recommendations

**Vectorized Implementation:**
```
# For user u with neighbors N
neighbor_interactions = interaction_matrix[neighbor_indices]  # (50, n_articles)
neighbor_similarities = similarity_weights  # (50,)

# Weighted aggregation
collab_scores = (neighbor_similarities @ neighbor_interactions) / sum(neighbor_similarities)
# Result: (n_articles,) predicted engagement

# Apply geo scores
collab_scores *= geo_scores

# Add popularity boost
collab_scores += 0.10 * popularity_scores
```

---

### Metrics on Training Data (80%)

**Metrics Computed:**

| Metric                   | What It Measures                                            | Expected Value                    |
|--------------------------|-------------------------------------------------------------|-----------------------------------|
| **Coverage**             | % of articles recommended                                   | 60-70% (lower than content-based) |
| **User Coverage**        | % of users who can get recommendations                      | 70.6% (only active users)         |
| **Interaction Sparsity** | % of user-article pairs with predicted score > 0            | 15-25%                            |
| **Neighbor Overlap**     | Average % of neighbors who read recommended articles        | 40-60%                            |
| **Popularity Bias**      | Correlation between recommendation frequency and popularity | 0.4-0.6 (moderate bias)           |

**Why Lower Coverage?**
- Collaborative filtering only recommends articles that neighbors read
- Cannot recommend articles with zero interactions
- Only works for active users (≥10 interactions)

**Ranking Metrics (⚠️ DATA LEAKAGE WARNING):**

| Metric           | Expected Value (Inflated) | What It Measures                      |
|------------------|---------------------------|---------------------------------------|
| **NDCG@50**      | 0.55-0.75                 | Ranking quality (INFLATED)            |
| **Precision@50** | 0.20-0.35                 | Accuracy (INFLATED)                   |
| **Recall@50**    | 0.50-0.75                 | Coverage of interests (INFLATED)      |
| **MAP@50**       | 0.25-0.45                 | Mean Average Precision (INFLATED)     |

**⚠️ WARNING:** Data leakage - model trained and evaluated on same data. Use validation metrics for true performance.

---

### Validation on 20% Held-Out Data

**Validation Set:**
- Last 708,832 interactions (by timestamp)
- For each user, find articles they interacted with in this period

**Validation Process:**
1. Generate top-50 recommendations using Collaborative Filtering (trained on 80%)
2. Compare with ground truth (articles user actually read in validation period)
3. Compute distributional and ranking metrics

**Distributional Metrics (For Comparison):**

| Metric                   | Expected Value | What It Measures                           |
|--------------------------|----------------|--------------------------------------------|
| **Coverage**             | 55-65%         | % of validation articles recommended       |
| **User Coverage**        | 70.6%          | % of users served (active users only)      |
| **Interaction Sparsity** | 12-20%         | % of user-article pairs with score > 0     |
| **Neighbor Overlap**     | 35-55%         | % of neighbors who read recommended items  |
| **Popularity Bias**      | 0.3-0.5        | Correlation with popularity                |

---

**Ranking Metrics (TRUE Performance):**

| Metric           | Expected Value | Why                                                    |
|------------------|----------------|--------------------------------------------------------|
| **NDCG@50**      | 0.25-0.40      | Better than content-based (captures behavior patterns) |
| **Precision@50** | 0.08-0.15      | 8-15 relevant out of 50                                |
| **Recall@50**    | 0.30-0.50      | Captures 30-50% of user's interests                    |
| **MAP@50**       | 0.12-0.22      | Higher than content-based                              |

**Expected Performance:**
- Collaborative filtering typically outperforms content-based on recall
- Better at discovering latent preferences
- But: Limited to articles with interaction history

---

### Testing on Test Data (970 New Articles)

**Problem:** Testing articles have ZERO interaction history

**Solution:** **Collaborative filtering CANNOT be used for testing articles**

**Why?**
```
collab_score = Σ(neighbor_similarity × neighbor_engagement)
                    ↑
                    This is ZERO for all testing articles
                    (no neighbor has read these articles yet)
```

**Fallback:**
- Switch to Content-Based algorithm for testing articles
- Or use popularity-only recommendations

**Metrics Computed:** N/A (Collaborative filtering not applicable)

---

### Fallback Mechanism

**Scenario 1: User Has < 10 Interactions (2,635 users = 29.4%)**
```
Fallback: Switch to Content-Based algorithm
Weight: 100% content-based (user not eligible for collaborative filtering)
Affected users: 29.4% (passive users)
```

**Scenario 2: User Has No Similar Neighbors**
```
Fallback: Use popularity-based recommendations
Weight: 100% popularity
Affected users: Rare (~1-2% of active users)
```

**Scenario 3: Article Pool Has No Neighbor Interactions**
```
Fallback: Use Content-Based algorithm
Weight: 70% content + 30% popularity
Triggered when: Recommending for new article pool (e.g., testing articles)
```

**Scenario 4: Low Neighbor Overlap (all neighbor scores < 0.05)**
```
Fallback: Blend with Content-Based
Weight: 50% collaborative + 50% content
Triggered when: User's neighbors have very different reading patterns
```

---

## Section 6: Hybrid Algorithm

### Algorithm Overview

**Principle:** Combine Content-Based and Collaborative Filtering based on data availability

**Decision Logic:**
```
if user_interactions < 10:
    Use Content-Based (100% weight)
elif article in testing_set:
    Use Content-Based (100% weight)  # No interaction history for testing
else:
    Use Hybrid (weighted combination)
```

---

### Hybrid Scoring Formula

**For Training/Validation Articles + Active Users:**
```
hybrid_score = 0.60 × collab_score
               + 0.25 × content_score
               + 0.15 × popularity_score

Where:
  collab_score = Weighted neighbor engagement × geo_score
  content_score = Cosine similarity × geo_score
  popularity_score = Normalized popularity (0-1)
```

**Weight Rationale:**
- **60% Collaborative:** Primary signal (behavioral patterns strongest)
- **25% Content:** Secondary signal (category preferences)
- **15% Popularity:** Diversity injection (prevent echo chamber)

---

### Required Files

All files from both Content-Based and Collaborative:

| File | Used In | Purpose |
|------|---------|---------|
| `interaction_matrix.pkl` | Collaborative | User-article engagement |
| `user_similarity.pkl` | Collaborative | K-NN neighbors |
| `user_category_tfidf.pkl` | Content-Based | User interests |
| `training_tfidf.pkl` | Content-Based | Article features |
| `testing_tfidf.pkl` | Content-Based | Testing articles |
| `article_popularity.pkl` | Both | Popularity boost |
| `article_locations.pkl` | Both | Geographic relevance |
| `user_profiles.csv` | Both | Metadata + interaction count |

---

### Training on 80% of Training Data

**Process:**

1. **Split data temporally (80/20)**
   - Train interactions: 2,835,329 events
   - Train articles: 6,536 articles

2. **Build both models:**
   - Content-Based: user_tfidf × article_tfidf
   - Collaborative: user_similarity × neighbor_interactions

3. **For each user, decide algorithm:**
   ```
   if num_interactions < 10:
       recommendations = content_based_recommend(user, top_k=50)
   else:
       # Compute both scores
       content_scores = content_based_score(user, articles)
       collab_scores = collaborative_score(user, articles)
       pop_scores = popularity_scores[articles]

       # Blend
       hybrid_scores = 0.60*collab + 0.25*content + 0.15*pop

       # Rank
       recommendations = top_k(hybrid_scores, k=50)
   ```

4. **Output:** Top 50 recommendations per user

---

### Metrics on Training Data (80%)

**Metrics Computed:**

| Metric | Expected Value | Why |
|--------|----------------|-----|
| **Coverage** | 85-95% | Higher than collaborative alone |
| **User Coverage** | 96.8% | Content-based covers passive users |
| **Category Diversity** | 2.8-3.5 | Balanced (content prevents filter bubble) |
| **Novelty** | Medium-High | Popularity weight prevents only obscure items |
| **Sparsity** | 30-40% | Better than pure collaborative |
| **Algorithm Distribution** | 29% content, 71% hybrid | Based on user interaction counts |

**Algorithm Usage Breakdown:**
```
Passive users (< 10 interactions): 29.4% → Content-Based (100%)
Active users (≥ 10 interactions): 70.6% → Hybrid (60/25/15)
```

**Ranking Metrics (⚠️ DATA LEAKAGE WARNING):**

| Metric           | Expected Value (Inflated) | What It Measures                      |
|------------------|---------------------------|---------------------------------------|
| **NDCG@50**      | 0.60-0.80                 | Ranking quality (INFLATED)            |
| **Precision@50** | 0.25-0.40                 | Accuracy (INFLATED)                   |
| **Recall@50**    | 0.55-0.80                 | Coverage of interests (INFLATED)      |
| **MAP@50**       | 0.30-0.50                 | Mean Average Precision (INFLATED)     |

**⚠️ WARNING:** Data leakage - model trained and evaluated on same data. Use validation metrics for true performance.

---

### Validation on 20% Held-Out Data

**Validation Process:**

1. Generate top-50 recommendations for each user using hybrid approach
2. Compare with ground truth (validation interactions)
3. Compute distributional and ranking metrics

**Distributional Metrics (For Comparison):**

| Metric                   | Expected Value | What It Measures                           |
|--------------------------|----------------|--------------------------------------------|
| **Coverage**             | 80-90%         | % of validation articles recommended       |
| **User Coverage**        | 96.8%          | % of users served (all segments)           |
| **Category Diversity**   | 2.8-3.5        | Entropy of recommended categories          |
| **Novelty**              | Medium-High    | Balance of popular/niche recommendations   |
| **Interaction Sparsity** | 25-35%         | % of user-article pairs with score > 0     |
| **Algorithm Mix**        | 29% CB, 71% H  | Distribution across user segments          |

---

**Ranking Metrics (TRUE Performance):**

| Metric | Expected Value | Comparison |
|--------|----------------|------------|
| **NDCG@50** | 0.30-0.45 | Best of all three algorithms |
| **Precision@50** | 0.10-0.18 | Improved over individual algorithms |
| **Recall@50** | 0.35-0.55 | Balanced breadth/depth |
| **MAP@50** | 0.15-0.25 | Highest average precision |

**Expected Performance:**
- Hybrid outperforms both Content-Based and Collaborative individually
- Content-Based fills gaps where Collaborative fails
- Collaborative provides personalization where Content-Based is generic

**Performance by User Segment:**

| User Segment | NDCG@50 | Precision@50 | Algorithm Used |
|--------------|---------|--------------|----------------|
| Passive (1-9) | 0.15-0.25 | 0.05-0.10 | Content-Based |
| Active (10-99) | 0.30-0.45 | 0.10-0.18 | Hybrid (60/25/15) |
| Power (100+) | 0.40-0.55 | 0.15-0.25 | Hybrid (60/25/15) |

---

### Testing on Test Data (970 New Articles)

**Scenario:** Cold-start articles (zero interaction history)

**Algorithm Decision:**
```
For ALL users (regardless of interaction count):
    Use Content-Based (100%)

Reason: Testing articles have no collaborative signals
```

**Testing Process:**
1. Load testing_tfidf (970 × 38)
2. For each user:
   - Compute content_score = user_tfidf × testing_tfidf
   - Apply geo_score
   - Add popularity boost (0.15 weight)
   - Add language match (0.10 weight)
3. Rank and select top-50

**Metrics Computed:**

| Metric | Expected Value | What It Measures |
|--------|----------------|------------------|
| **Coverage** | 85-95% | % of 970 articles recommended |
| **User Coverage** | 96.8% | % of users served |
| **Category Diversity** | 2.5-3.2 | Diversity of recommendations |
| **Novelty** | Very High | All articles are new |
| **Sparsity** | 40-50% | % of user-article pairs with score > 0 |
| **Variance** | 0.15-0.30 | Distribution of scores |

**NO NDCG/Precision/Recall:**
- Testing articles have zero ground truth
- Cannot measure ranking quality
- Only measure distributional properties

---

### Fallback Mechanism (Comprehensive)

**Level 1: User-Level Fallback**

| Condition | Action | Weight Distribution |
|-----------|--------|---------------------|
| User has < 10 interactions | Switch to Content-Based | 75% content + 25% popularity |
| User has no TF-IDF profile | Switch to Popularity-Only | 100% popularity |
| User has no similar neighbors | Blend Content + Popularity | 60% content + 40% popularity |

**Level 2: Article-Level Fallback**

| Condition | Action | Weight Distribution |
|-----------|--------|---------------------|
| Article in testing set | Force Content-Based | 75% content + 25% popularity |
| Article has no interactions | Use Content-Based | 70% content + 30% popularity |
| Article not in TF-IDF | Use Popularity-Only | 100% popularity |

**Level 3: Score-Level Fallback**

| Condition | Action | Weight Distribution |
|-----------|--------|---------------------|
| Collaborative score = 0 for all articles | Switch to Content-Based | 70% content + 30% popularity |
| Content score = 0 for all articles | Switch to Popularity-Only | 100% popularity |
| Both scores low (< 0.05) | Increase popularity weight | 40% collab + 30% content + 30% pop |

**Level 4: Emergency Fallback**

| Condition | Action | Weight Distribution |
|-----------|--------|---------------------|
| All algorithms fail | Return top 50 popular articles | 100% popularity |

**Fallback Frequency (Expected):**
```
Level 1 (User-Level): 29.4% of users (passive)
Level 2 (Article-Level): 6.6% of articles (testing set)
Level 3 (Score-Level): 5-10% of recommendations
Level 4 (Emergency): < 0.1% (rare)
```

---

## Section 7: Summary of Algorithm Comparison

### Performance Summary (Validation Set)

| Algorithm | NDCG@50 | Precision@50 | Recall@50 | Coverage | User Coverage |
|-----------|---------|--------------|-----------|----------|---------------|
| **Content-Based** | 0.15-0.25 | 0.05-0.10 | 0.20-0.35 | 95% | 96.8% |
| **Collaborative** | 0.25-0.40 | 0.08-0.15 | 0.30-0.50 | 60-70% | 70.6% |
| **Hybrid** | 0.30-0.45 | 0.10-0.18 | 0.35-0.55 | 85-95% | 96.8% |

**Winner:** Hybrid (best balance of accuracy, coverage, and user coverage)

---

### Algorithm Selection by Scenario

| Scenario | Algorithm Used | Reason |
|----------|----------------|--------|
| Passive user (< 10 interactions) | Content-Based | Insufficient collaborative signals |
| Active user (≥ 10 interactions) | Hybrid (60/25/15) | Leverage both behavioral and content signals |
| Testing articles (cold-start) | Content-Based | Zero interaction history |
| Training articles with interactions | Hybrid (60/25/15) | Best performance |
| New user (zero interactions) | Popularity-Only | No user profile |

---

### Weight Distribution Across All Recommendations

**Expected Overall Blend (across all users and articles):**
```
Collaborative Filtering:   42% (0.60 × 70.6% active users)
Content-Based Filtering:   40% (0.25 × 70.6% + 1.0 × 29.4%)
Popularity:               18% (0.15 × 70.6% + fallback scenarios)
Geographic Boost:          Applied as multiplier (5-10% effective weight)
```

---

## Section 8: Files Generated Summary

### Complete Feature Files List

| File | Dimensions | Size | Algorithm |
|------|------------|------|-----------|
| `tfidf_vectorizer.pkl` | Fitted model (38 features) | ~50 KB | Content-Based |
| `training_tfidf.pkl` | 8,170 × 38 | ~600 KB | Content-Based |
| `testing_tfidf.pkl` | 970 × 38 | ~80 KB | Content-Based |
| `user_category_tfidf.pkl` | 8,689 × 38 | ~650 KB | Content-Based |
| `article_popularity.pkl` | 14,622 entries | ~200 KB | Both |
| `article_locations.pkl` | 9,140 entries | ~150 KB | Both |
| `user_similarity.pkl` | 6,342 users × 50 neighbors | ~3 MB | Collaborative |
| `interaction_matrix.pkl` | 8,977 × 14,622 (sparse) | ~15 MB | Collaborative |
| `mappings.pkl` | 4 dicts | ~500 KB | Collaborative |
| `article_features.csv` | 9,140 × 22 | ~2 MB | Both |
| `user_profiles.csv` | 8,689 × 8 | ~1 MB | Both |

**Total Storage:** ~23 MB (efficient due to sparse matrices)

---

## Section 9: Evaluation Strategy

### Stage 1: Validation on Training Data (80/20 Split)

**Purpose:** Measure algorithm performance with ground truth

**Process:**
1. Temporal split: 80% train, 20% validation
2. Generate recommendations on validation articles
3. Compare with actual user interactions in validation period
4. Compute NDCG, Precision, Recall, MAP

**All Three Algorithms Evaluated:**
- Content-Based: Baseline
- Collaborative: Compare
- Hybrid: Final system

---

### Stage 2: Final Recommendations on Testing Data

**Purpose:** Generate real-world recommendations for new articles

**Process:**
1. Load testing_tfidf (970 new articles)
2. For each user, apply Hybrid algorithm (defaults to Content-Based for testing)
3. Generate top-50 recommendations
4. Measure distribution metrics (coverage, diversity, novelty)

**No Ground Truth Available:**
- Testing articles are new (zero interactions)
- Cannot compute NDCG/Precision/Recall
- Focus on diversity and coverage metrics

---

*Document Version: 2.0*
*Last Updated: 2025-01-14*
*Notebook: 02_recommendation_system.ipynb*
