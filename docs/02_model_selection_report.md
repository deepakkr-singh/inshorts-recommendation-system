# Model Selection Rationale

**Data-Driven Algorithm Choice Based on EDA Findings**

This document explains why Content-Based Filtering and Collaborative Filtering were selected as complementary approaches for the news recommendation system. The decision was driven directly by empirical insights from Exploratory Data Analysis (EDA).



## Summary

**Algorithms Selected:**
- **Content-Based Filtering** to Handles cold-start, new articles, passive users
- **Collaborative Filtering** to Leverages interaction patterns for active users
- **Hybrid Strategy** to Combines both to address data sparsity and heterogeneity

**Key Finding:** No single algorithm sufficiently addresses all observed challenges. A Hybrid approach is necessary given the observed data characteristics.



## 1. Cold-Start Problem (New Articles)

### EDA Finding

| Metric | Value |
|------------------------------|-----------------|
| **Testing articles** | 970 |
| **Overlap with training** | 0% (0 articles) |
| **User interaction history** | None |

**Impact:** 970 new articles have zero historical engagement data.

### Algorithm Implication

```
Collaborative Filtering:
 - Requires interaction history
 - Cannot recommend articles with zero user events

Content-Based Filtering:
 - Uses article metadata (category, language, type)
 - Matches to user interest profiles
 - Works immediately for new articles
```

### Decision

**Content-Based Filtering is mandatory** for item cold-start scenarios.

**Implementation:**
- Uses TF-IDF vectors (~32–35 category features after cleaning and min_df=2 filtering)
- Cosine similarity between article vectors and user interest profiles
- Files: `testing_tfidf.pkl`, `user_category_tfidf.pkl`



## 2. User Segmentation by Activity Level

### EDA Finding

| User Segment | Interactions | Percentage | Count |
|-------------------|--------------|------------|-------|
| **Passive users** | 1–9 | 22.87% | 2,053 |
| **Active users** | 10–99 | 37.31% | 3,349 |
| **Power users** | 100+ | 39.82% | 3,575 |

**Total active users:** 8,977

### Algorithm Implication

**Passive Users (22.87%):**
- Insufficient interaction history for reliable collaborative filtering
- Risk of noisy neighbor recommendations
- Need content-based fallback

**Active + Power Users (77.13%):**
- Rich interaction data (10–1000+ events per user)
- Collaborative filtering can discover latent preferences
- User-user similarity is meaningful

### Decision

```
if user_interactions < 10:
 use Content-Based Filtering
else:
 use Collaborative Filtering (with Content-Based fallback)
```

**Implementation:**
- User profiles track `num_interactions` (in `user_profiles.csv`)
- Recommender checks interaction count before choosing algorithm
- Files: `user_similarity.pkl` (only for active users)



## 3. Interaction Data Sparsity

### EDA Finding

| Metric | Value |
|------------------------------|-------------------------------|
| **Interaction matrix shape** | 8,977 users × 14,622 articles |
| **Non-zero entries** | 3,544,161 (2.70%) |
| **Sparsity** | 97.30% |

**Interpretation:** Most users interacted with <3% of articles.

### Algorithm Implication

**Collaborative Filtering Alone:**
- Likely to struggle under extreme sparsity
- May recommend only popular articles
- Limited coverage of long-tail content

**Content-Based Filtering:**
- Unaffected by sparsity
- Can recommend niche articles matching user interests
- Ensures diversity in recommendations

### Decision

**Hybrid approach mitigates sparsity-related limitations:**
- Collaborative filtering for popular/trending content
- Content-based filtering for personalized long-tail recommendations

**Implementation:**
- Sparse CSR matrix format (`interaction_matrix.pkl`)
- Content-based as fallback when collaborative similarity is low
- Files: `interaction_matrix.pkl` (97.30% sparse)



## 4. Availability of Rich Metadata

### EDA Finding

| Feature | Coverage | Quality |
|-------------------------------|---------------|---------------------------------|
| **Article category** | 99.69% | 56 unique categories |
| **Article language** | 98.77% | English, Hindi, Malayalam, etc. |
| **Article type** | 96.19% | news, video_news |
| **User category preferences** | 96.8% | Computed from reading history |
| **TF-IDF features** | 38 dimensions | After min_df=2 filtering |

**Observation:** Users exhibit clear category preference patterns.

### Algorithm Implication

**Content-Based Filtering:**
- Can reliably match users and articles using metadata
- TF-IDF captures category importance (e.g., "sports", "technology")
- Works even without interaction data

**Example:**
```
User A reads: [sports, sports, cricket, football]
Article B categories: [sports, football]
→ High cosine similarity → Recommend
```

### Decision

**Metadata quality strongly supports Content-Based Filtering.**

**Implementation:**
- TF-IDF vectorization on category text (38 features)
- User profiles aggregated with engagement weights:
 - TimeSpent-Front: 1.0
 - TimeSpent-Back: 2.0
 - Bookmarked: 3.0
 - Shared: 5.0
- Files: `tfidf_vectorizer.pkl`, `training_tfidf.pkl`, `testing_tfidf.pkl`, `user_category_tfidf.pkl`



## 5. Strength of Interaction Signals

### EDA Finding

| Metric | Value |
|---------------------------------|--------------------------------------------|
| **Total interaction events** | 3,544,161 |
| **Users with ≥10 interactions** | 6,924 (77.1%) |
| **Engagement types** | TimeSpent (Front/Back), Bookmarked, Shared |
| **Implicit feedback** | Multiple signals (time, bookmarks, shares) |

**Observation:** Rich engagement data reveals user preferences.

### Algorithm Implication

**Collaborative Filtering:**
- Sufficient data to compute meaningful user similarity
- Can discover latent preferences (e.g., users who read A also read B)
- Implicit feedback signals indicate interest strength

**Example:**
```
User X: Bookmarked articles [123, 456, 789]
User Y: Bookmarked articles [123, 456, 999]
→ High similarity → Recommend 789 to User Y
```

### Decision

**Collaborative Filtering becomes viable and effective** for users with sufficient interaction history.

**Implementation:**
- User-user cosine similarity on interaction matrix
- K=50 nearest neighbors per user
- Weighted scoring by neighbor similarity
- Files: `user_similarity.pkl` (6,924 users with neighbors)



## Decision Matrix: EDA Insights to Algorithm Choice

| EDA Insight | Content-Based | Collaborative | Hybrid Outcome |
|------------------------------------|---------------|---------------|------------------------|
| **New articles (0% overlap)** | Works | Fails | Content-Based used |
| **Passive users (22.87%)** | Works | ~ Noisy | Content-Based fallback |
| **Active users (77.13%)** | Works | Works | Combined scores |
| **Category metadata (99.69%)** | Required | - Not used | Leveraged |
| **Interaction data (3.5M events)** | - Not used | Required | Leveraged |
| **Extreme sparsity (97.30%)** | Robust | ~ Limited | Complementary |

**Legend:**
- Effective
- Ineffective
- ~ Partially effective
- − Not applicable



## Implementation Summary

### Files Generated (Section 1: Feature Engineering)

| File | Purpose | Dimensions |
|---------------------------|---------------------------|-----------------------------|
| `tfidf_vectorizer.pkl` | Fitted TF-IDF model | 38 features |
| `training_tfidf.pkl` | Training article vectors | 8,170 × 38 |
| `testing_tfidf.pkl` | Testing article vectors | 970 × 38 |
| `user_category_tfidf.pkl` | User interest vectors | 8,689 × 38 |
| `article_popularity.pkl` | Popularity scores | 14,622 articles |
| `article_locations.pkl` | Inferred locations | 9,140 articles |
| `user_similarity.pkl` | K-NN neighbors | 6,924 users |
| `interaction_matrix.pkl` | User-article engagement | 8,977 × 14,622 (sparse) |
| `mappings.pkl` | ID mappings | user_to_idx, article_to_idx |
| `article_features.csv` | Combined article features | 9,140 rows |
| `user_profiles.csv` | Combined user profiles | 8,689 rows |


### Evaluation Strategy

**Two-Stage Approach:**

1. **Stage 1 (Validation on Training Data):**
 - Temporal 80/20 split (by timestamp)
 - Evaluate on 20% held-out training articles
 - Metrics: NDCG@50, Precision@50, Recall@50, MAP@50
 - Both algorithms tested with ground truth

2. **Stage 2 (Final Recommendations on Testing Data):**
 - Generate recommendations for 970 new articles
 - Content-Based only (no interaction history)
 - Metrics: Coverage, Diversity, Novelty
 - No ground truth available




