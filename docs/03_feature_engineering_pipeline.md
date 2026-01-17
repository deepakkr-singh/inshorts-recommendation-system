# Inshorts News Recommendation System - Technical Documentation

## Project Overview
This project implements a hybrid recommendation system for Inshorts news articles, combining content-based filtering, collaborative filtering, and hybrid approaches to recommend personalized news content to users.

---

## Data Architecture

## Category Similarity (TF-IDF Feature Engineering)

### HashId Connection Model
All data is connected through `hashId` (Article ID) as the primary key:

```
┌─────────────────────────────────────────────────────────┐
│ ARTICLE DATA                                            │
├─────────────────────────────────────────────────────────┤
│ training_content.csv → 8,170 articles (8,168 unique)    │
│ testing_content.csv → 970 articles (970 unique)         │
│ article_features.csv → 9,140 articles (9,138 unique)    │
│ = train (8,168) + test (970) = 9,138 total              │
└─────────────────────────────────────────────────────────┘
                           ↓ (hashId)
┌─────────────────────────────────────────────────────────┐
│ FEATURE MAPPINGS                                        │
├─────────────────────────────────────────────────────────┤
│ training_tfidf.pkl → 8,170 × 31 (articles × cats)       │
│ testing_tfidf.pkl → 970 × 31 (articles × cats)          │
│ user_category_tfidf.pkl → 8,689 × 31 (users × cats)     │
│ 8,689 unique users from events.csv with category data   │
│ (288 users excluded: only read articles without         │
│ 'categories' field - cannot build TF-IDF profiles)      │
└─────────────────────────────────────────────────────────┘
                           ↑ (hashId)
┌─────────────────────────────────────────────────────────┐
│ USER INTERACTIONS                                       │
├─────────────────────────────────────────────────────────┤
│ events.csv → 3.5M interactions                          │
│ Unique articles: 14,622 (includes historical)           │
│ Unique users: 8,977 (incl. 288 no category user)        │
│                                                         │
│ Article Distribution:                                   │
│ - 8,155 articles in TRAINING (99.84% of train set)      │
│ - 0 articles in TESTING (0% - cold start scenario)      │
│ - 6,468 historical articles (removed/not in dataset)    │
└─────────────────────────────────────────────────────────┘
```

---

## Data Quality Summary

### Completeness Check

| Feature | Training | Testing | Coverage |
|--------------------|----------|---------|-----------|
| **hashId** | 100% | 100% | Complete |
| **categories** | 99.69% | 99.69% | Excellent |
| **newsLanguage** | 98.67% | 99.69% | Excellent |
| **newsType** | 95.75% | 99.90% | Very Good |
| **newsDistrict** | 0.40% | 32.27% | Extremely Sparse |
| **TF-IDF vectors** | 99.66% | 99.59% | Excellent |

### User Data Quality

| Feature | Coverage | Coverage |
|-------------------------|--------------------|-------------------------------------------|
| **deviceId** | 100% (8,689 users) | Complete |
| **category_text** | 100% | Complete (All users have reading history) |
| **language_preference** | 7.87% | Extremely Sparse |
| **user_city** | 7.87% | Extremely Sparse |
| **preferred_newsType** | 100% | Complete (mostly 'NEWS') |
| **segment** | 100% | Complete |

---

## 1. Category Similarity (50% weight)
- **Source**: TF-IDF vectors from `categories` column
- **User**: `user_category_tfidf.pkl` (8,689 users × 31 features)
- **Article**: `training_tfidf.pkl` or `testing_tfidf.pkl`
- **Method**: Cosine similarity between user and article TF-IDF vectors

### 2. Source Data

#### Training Content
- **File**: `assignment/data/processed/training_content.csv`
- **Shape**: 8,170 rows × 12 columns
- **Key Column**: `categories` (comma-separated category strings)
- **Columns**: `['hashid', 'title', 'content', 'newsType', 'author', 'categories', 'hashtags', 'newsDistrict', 'createdAt', 'updatedAt', 'newsLanguage', 'sourceName']`

**Category Statistics:**
- Total articles: **8,170**
- Missing categories: **25 (0.31%)**
- Articles with categories: **8,145**
- Unique categories: **55**
- Total category mentions: **10,064**

#### Testing Content
- **File**: `assignment/data/processed/testing_content.csv`
- **Shape**: 970 rows × 12 columns
- **Key Column**: `categories` (comma-separated category strings)
- **Columns**: Same as training

**Category Statistics:**
- Total articles: **970**
- Missing categories: **3 (0.31%)**
- Articles with categories: **967**
- Unique categories: **19**
- Total category mentions: **1,067**

---

### 3. TF-IDF Transformation

#### Training TF-IDF Matrix
- **File**: `assignment/data/features/training_tfidf.pkl`
- **Original Shape**: 8,170 articles × 12 columns
- **TF-IDF Shape**: **8,170 articles × 31 category features**
- **Type**: `scipy.sparse.csr_matrix` (Compressed Sparse Row)
- **Non-zero elements**: 10,039
- **Sparsity**: **96.04%**
- **Zero vectors**: 30 articles (0.37%)

#### Testing TF-IDF Matrix
- **File**: `assignment/data/features/testing_tfidf.pkl`
- **Original Shape**: 970 articles × 12 columns
- **TF-IDF Shape**: **970 articles × 31 category features**
- **Type**: `scipy.sparse.csr_matrix` (Compressed Sparse Row)
- **Non-zero elements**: 1,066
- **Sparsity**: **96.45%**
- **Zero vectors**: 4 articles (0.41%)

#### TF-IDF Vectorizer
- **File**: `assignment/data/features/tfidf_vectorizer.pkl`
- **Type**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Vocabulary Size**: **31 categories**
- **Fit on**: Training data categories only
- **Transform**: Applied to both training and testing data

#### Key Decisions
- **Lowercasing (`lowercase=True`)**: Ensures case-insensitive matching (e.g., "VIDEO_NEWS" to "video_news")
- **Minimum Document Frequency (`min_df=2`)**: Filters out 21 rare categories appearing in only 1 article (likely noise/errors)

#### min_df Impact Analysis

| min_df | Features | Density | Dropped | Avg/Article | Zero Vectors |
|----------|------------|-----------|-----------|---------------|----------------|
| 1 | 52 | 2.37% | 0 | 1.23 | 29 (0.35%) |
| 2 | 31 | 3.96% | 21 | 1.23 | 30 (0.37%) |
| 5 | 24 | 5.11% | 28 | 1.23 | 32 (0.39%) |
| 10 | 22 | 5.57% | 30 | 1.22 | 32 (0.39%) |

**Column Explanations:**
- **Features**: Number of category features in vocabulary after filtering
- **Density**: Percentage of non-zero elements in TF-IDF matrix (higher = less sparse)
- **Dropped**: Number of categories removed due to min_df threshold
- **Avg/Article**: Average non-zero features per article (information content per article)
- **Zero Vectors**: Articles with no categories in vocabulary (all categories filtered out)

**Why min_df=2 is optimal:**
- Removes 21 noise categories (appeared once = likely errors: "₹500", "facts", "question")
- Balances information retention vs dimensionality reduction
- Minimal impact: avg 1.23 features/article maintained (same as min_df=1)
- Only 1 additional zero vector (30 vs 29) - 0.02% impact

---

### 4. Missing Category Handling

#### Training Data
- **Articles with missing categories**: 25 (0.31%)
- **Articles with zero TF-IDF vectors**: 30 (0.37%)
- **Example indices**: 367, 368, 1268, 1275, 1760
- **TF-IDF values**: All zeros (sum = 0.000000)

#### Testing Data
- **Articles with missing categories**: 3 (0.31%)
- **Articles with zero TF-IDF vectors**: 4 (0.41%)
- **Example indices**: 678, 679, 705
- **TF-IDF values**: All zeros (sum = 0.000000)

#### Natural Handling (Implicit Fallback)
```python
# Articles with zero category vectors automatically get category_similarity = 0
# System naturally falls back to other features (50% of total score)

if article_has_zero_categories:
    category_similarity = 0.0 # Zero vector → 0 cosine similarity
    content_score = (
        0.0 * 0.50 + # Category: 0 (missing)
        lang_score * 0.15 + # Still contributes
        popularity * 0.15 + # Still contributes
        type_score * 0.10 + # Still contributes
        geo_score * 0.10 # Still contributes
    )
    # Max possible score: 0.50 (if all other features match perfectly)
```

#### Impact Assessment

**Overall Impact:**
- **Total articles affected**: 34 / 9,140 (0.37%)
- **Users affected**: 0 / 8,689 (0.00%)

**Recommendation: No explicit fallback mechanism is required under the observed data conditions.**

**Why NO fallback is required:**

1. **Minimal Impact**: Only 0.37% of articles have zero category vectors - negligible edge case
2. **Implicit Fallback Exists**: Scoring formula naturally handles this
   - Other 4 features contribute 50% of score (language 15%, popularity 15%, type 10%, geo 10%)
   - Articles still get recommended based on metadata matching
3. **Design Rationale**: Zero category vectors occur when:
   - Article has rare/unknown categories to Should rely on popularity/language - Cold start scenario to Metadata becomes more important - This behavior **aligns with production system requirements** for graceful degradation
4. **No User Impact**: All users (100%) have valid category profiles from reading history

**When would explicit fallback be needed?**
- If **>5% of articles** had zero vectors to Assign default category (e.g., "miscellaneous")
- If **>10% of users** had zero vectors to Use popularity-based recommendations
- If category weight was **>70%** to Zero categories would severely hurt recommendations
- Current system: **0.37% articles, 0% users, 50% weight** to Natural handling is sufficient

**Note**: While category fallback is unnecessary, **user metadata fallback IS critical** (92% missing language/location - see User Profile section)

---

### 5. Column Names Reference

### Original CSV Files (Training & Testing)
Both `training_content.csv` and `testing_content.csv` contain:

```python
columns = [
    'hashid', # Primary key (article ID)
    'title', # Article headline
    'content', # Article text content
    'newsType', # NEWS, VIDEO_NEWS, etc.
    'author', # Author ID
    'categories', # Used for TF-IDF (comma-separated)
    'hashtags', # Article hashtags
    'newsDistrict', # Geographic location code
    'createdAt', # Publication timestamp
    'updatedAt', # Last update timestamp
    'newsLanguage', # english, hindi, telugu, etc.
    'sourceName' # Content source
]
```

### Feature Files

#### article_features.csv
```python
columns = [
    'hashid', # Primary key
    'newsLanguage', # Used for language matching
    'newsType', # Used for type matching
    'popularity', # Used for popularity scoring
    'is_test', # Train/test flag
    'lang_ANI', # One-hot encoded language
    'lang_Twitter',
    'lang_english',
    'lang_gujarati',
    'lang_hindi',
    'lang_kannada',
    'lang_telugu',
    'lang_भाषा',
    'newsType_encoded' # Encoded news type
]
```

#### user_profiles.csv
```python
columns = [
    'deviceId', # Primary key (user ID)
    'category_text', # User reading history categories
    'num_interactions', # Total read count
    'language_preference', # User language (92% missing!)
    'user_city', # User location (92% missing!)
    'preferred_newsType', # User type preference
    'segment' # User engagement segment
]
```

## 2. Popularity Score (15% Weight)

### 1. Overview
- **Weight**: 15% of total content-based score
- **Source**: Aggregated from `events.csv` (3.5M user interactions)
- **Method**: Logarithmic scoring combining unique users and total events
- **Coverage**: Training 99.84%, Testing 0% (cold start by design)
- **Implementation**: Precomputed dictionary for O(1) lookup
- **Key Characteristic**: **Article-only metric** (NOT user-personalized)
  - Category Similarity: User-specific (each user has different category preferences)
  - Language Match: User-specific (each user prefers different languages)
  - **Popularity: Global** (all users see same popularity score for each article)

---

### 2. Source Data

#### Events Dataset
- **File**: `assignment/data/processed/events.csv`
- **Total Interactions**: ~3.5 million events
- **Unique Articles**: 14,622 (includes historical articles)
- **Unique Users**: 8,977
- **Event Types**: TimeSpent-Front, TimeSpent-Back, Bookmarked, Shared

**Article Distribution in Events:**

| Article Set | Count | Percentage | In Events |
|-------------------------|--------|------------|-------------------------|
| **Training articles** | 8,155 | 99.84% | Yes (Have interactions) |
| **Testing articles** | 0 | 0.00% | No (No interactions) |
| **Historical articles** | 6,468 | 44.22% | Yes (Have interactions) |
| **Total in events** | 14,622 | 100% | All |

**Key Observations:**
- **99.84% training articles** (8,155/8,170) have popularity data
- **0% testing articles** (0/970) have popularity data (cold start scenario)
- **Historical articles** (6,468) removed from dataset but still in events.csv

---

### 3. Popularity Computation Formula

#### Algorithm Overview
```python
# Step 1: Aggregate per article
popularity_stats = events.groupby('hashId').agg({
    'deviceId': 'nunique', # Count unique users (breadth)
    'event_type': 'count' # Count total events (depth)
}).rename(columns={
    'deviceId': 'unique_users',
    'event_type': 'total_events'
})

# Step 2: Calculate raw popularity score
popularity_stats['popularity_score'] = (
    np.log1p(popularity_stats['unique_users']) * 0.7 +
    np.log1p(popularity_stats['total_events']) * 0.3
)

# Step 3: Normalize to [0, 1]
max_pop = popularity_stats['popularity_score'].max()
popularity_stats['popularity_normalized'] = (
    popularity_stats['popularity_score'] / max_pop
)

# Step 4: Convert to dictionary
article_popularity = popularity_stats['popularity_normalized'].to_dict()
```

#### Formula Breakdown

**Raw Score:**
```
popularity_score = log₁₊(unique_users) × 0.7 + log₁₊(total_events) × 0.3
```

**Components:**
1. **Unique Users (70% weight)**:
   - Measures **breadth** of appeal (how many people engaged)
   - Higher weight = prioritize articles read by many users
   - `log₁₊(x) = log(1 + x)` prevents overflow and dampens outliers

2. **Total Events (30% weight)**:
   - Measures **depth** of engagement (repeat interactions)
   - Lower weight = secondary signal (quality > quantity)
   - Captures bookmarks, shares, multiple reads

**Normalization:**
```python
normalized = raw_score / max(raw_score)
# Result: [0.0, 1.0] range
```

---

### 4. Coverage Analysis

#### Training Articles (8,170 total)

| Status | Count | Percentage | Popularity |
|-------------------------|-------|------------|----------------|
| **Has popularity data** | 8,155 | 99.84% | [0.083, 1.000] |
| **Missing from events** | 13 | 0.16% | 0.0 (fallback) |
| **Total** | 8,170 | 100% | Complete |

**Why 13 missing?**
- Articles exist in `training_content.csv` but not in `events.csv`
- Likely test/seed data never actually shown to users
- Correctly assigned 0.0 popularity (no engagement = no popularity)

#### Testing Articles (970 total)

| Status | Count | Percentage | Popularity |
|----------------------------|-------|------------|-----------------|
| **Has popularity data** | 0 | 0.00%. | N/A |
| **Cold start (no events)** | 970 | 100% | 0.0 (by design) |
| **Total** | 970 | 100% | All zero |

**Why all testing = 0?**
- Testing articles are **future/unseen content** (not in historical events)
- This design choice **simulates real-world cold start scenarios**
- Recommendations must rely on other 85% of score (categories, language, type, geo)

#### Historical Articles (6,468)

| Status | Count | Percentage | Popularity |
|-------------------------|-------------|------------------------------|----------------|
| **In events.csv** | 6,468 | 44.22% of articles in events | [0.083, 1.000] |
| **In training/testing** | 0 | Not in content CSVs | Archived |
| **Status** | **Removed** | Old articles no longer used | Not used |

**What are Historical Articles?**
- Articles that appear in `events.csv` (have user interactions)
- But **NOT** in `training_content.csv` or `testing_content.csv`
- 6,468 out of 14,622 total articles in events = **44.22%**

**Why excluded?**
- Old/archived articles removed from current dataset
- Likely from earlier time periods before data collection cutoff
- Popularity computed but not used (articles don't exist in recommendation pool)
- Kept in `article_popularity.pkl` for completeness/reference

---

### 5. Popularity Distribution

#### Statistical Summary

```python
# From 14,622 articles with popularity data
Min: 0.083
25%: 0.165
Median: 0.643
75%: 0.687
Max: 1.000
Mean: 0.470
Std Dev: 0.264
```

**Distribution Characteristics:**
- **Right-skewed**: Most articles have low-medium popularity
- **Long tail**: Few viral articles (top 5% have popularity > 0.7)
- **No zeros in events**: Even least popular article has some engagement (min = 0.083)

#### Example Articles

**High Popularity (>0.9):**
- Many unique users + high total events
- Example: Breaking news, trending topics, viral content

**Medium Popularity (0.3-0.7):**
- Moderate user reach
- Example: Regular news articles, niche topics with engaged audience

**Low Popularity (0.08-0.3):**
- Few users or low engagement
- Example: Newly published, specialized content, low-visibility articles

---

### 6. Implementation Details

#### Storage Format

**Dictionary Structure:**
```python
article_popularity = {
    'hashid_1': 0.856, # High popularity
    'hashid_2': 0.234, # Medium popularity
    'hashid_3': 0.083, # Low popularity
    # ... 14,622 total entries
}
```

**File:**
- **Path**: `assignment/data/features/article_popularity.pkl`
- **Type**: `dict[str, float]`
- **Size**: ~14,622 entries
- **Memory**: ~500 KB (negligible)

#### Precomputation Strategy

Unlike language/type (precompute full matrix), popularity uses **on-the-fly lookup**:

```python
def _precompute_features(self):
    # Extract article hashids
    article_hashids = self.test_articles['hashid'].values
    # Shape: (970,)

    # Lookup popularity for each article (O(1) per lookup)
    self.pop_vector = np.array([
        self.popularity.get(hashid, 0.0) # Default 0.0 if missing
        for hashid in article_hashids
    ])
    # Result: (970,) array

    # Broadcasting during recommendation
    # (970,) → (batch_size, 970) via np.broadcast_to
```

**Why dictionary lookup (not matrix)?**
- **Sparse data**: Only 14,622 articles have popularity (not all possible articles)
- **Efficient memory**: Dict uses ~500 KB vs precomputed matrix would be ~64 MB
- **Fast lookup**: O(1) dictionary access
- **Easy fallback**: `.get(hashid, 0.0)` handles missing articles gracefully

---

### 7. Usage in Recommendation

#### Batch Processing

```python
def recommend_all(self, top_k=50, batch_size=3000):
    """Generate recommendations using popularity scores"""

    for batch_start in range(0, total_users, batch_size):
        user_indices = list(range(batch_start, batch_end))
        n_batch_users = len(user_indices)

        # 1. Category similarity (50%)
        category_sim = cosine_similarity(user_tfidf_batch, self.article_tfidf)

        # 2. Language scores (15%)
        lang_scores = self.lang_matrix[user_indices, :]

        # 3. Popularity scores (15%) - broadcast to batch
        pop_scores_matrix = np.broadcast_to(
            self.pop_vector, # (970,)
            (n_batch_users, 970) # → (3000, 970)
        )
        # Same popularity for all users (content-agnostic)

        # 4. Type preference (10%)
        type_scores = self.type_matrix[user_indices, :]

        # 5. Geographic relevance (10%)
        geo_scores = self.geo_matrix[user_indices, :]

        # Weighted combination
        content_scores = (
            category_sim * 0.50 +
            lang_scores * 0.15 +
            pop_scores_matrix * 0.15 + # ← Popularity contribution
            type_scores * 0.10 +
            geo_scores * 0.10
        )
```

#### Broadcasting Mechanism

**Popularity is user-agnostic** (same for everyone):

```python
# Input: (970,) vector
self.pop_vector = [0.856, 0.234, 0.083, ...]

# Broadcasting to batch
pop_scores_matrix = np.broadcast_to(self.pop_vector, (3000, 970))
# Shape: (3000 users, 970 articles)
# Memory: Efficient (no actual copy, just view)

# Result: All users see same article popularity
# User 0: [0.856, 0.234, 0.083, ...]
# User 1: [0.856, 0.234, 0.083, ...]
# User 2: [0.856, 0.234, 0.083, ...]
```

#### Example Scoring

**Scenario 1: Training Article (High Popularity)**
```python
article = training_article_1
popularity = 0.85 # 85% normalized score

# Contribution to final score
popularity_contribution = 0.85 × 0.15 = 0.1275

# Full score example (category match + language match + popularity)
content_score = 0.60*0.50 + 1.0*0.15 + 0.85*0.15 + 1.0*0.10 + 0.5*0.10
              = 0.30 + 0.15 + 0.1275 + 0.10 + 0.05
              = 0.7275 (strong recommendation)
```

**Scenario 2: Testing Article (Cold Start)**
```python
article = testing_article_1
popularity = 0.0 # No historical data

# Contribution to final score
popularity_contribution = 0.0 × 0.15 = 0.0

# Full score example (must rely on other features)
content_score = 0.60*0.50 + 1.0*0.15 + 0.0*0.15 + 1.0*0.10 + 0.5*0.10
              = 0.30 + 0.15 + 0.0 + 0.10 + 0.05
              = 0.60 (moderate recommendation - popularity handicap)
```

**Impact:**
- Testing articles lose 15% potential boost (max 0.15 from popularity)
- Must compensate with strong category/language match
- This behavior aligns with standard production practices, where new content initially relies on metadata rather than historical performance

---

### 8. Cold Start Problem

#### The Challenge

**Problem Statement:**
- Testing articles = future/unseen content
- Zero historical interactions to Zero popularity
- 15% of score permanently = 0

**Why This Happens:**
```python
events.csv:
  - Contains interactions BEFORE data collection cutoff
  - Training articles: 8,155 have history (99.84%)
  - Testing articles: 0 have history (0%)

Result:
  - Training recommendations: Popularity helps (0.08-1.0 boost)
  - Testing recommendations: No popularity boost (0.0)
```

#### Is This a Problem?

**Short Answer: No - This behavior is consistent with real-world cold-start deployment scenarios.**

**Reasons:**
1. **Real-world accuracy**: Simulates actual cold start (new articles have no history)
2. **Tests other features**: Forces system to rely on content matching (50%), language (15%), type (10%), geo (10%)
3. **Fair evaluation**: All testing articles start equal (no popularity bias)
4. **Expected in production**: New articles always start with 0 popularity

#### Alternative Approaches (Not Used)

**Option 1: Use Training Popularity as Proxy**
```python
# Assign average popularity to testing articles
testing_popularity = 0.470 # Mean from training
```
**Why rejected:**
- Creates artificial boost
- Doesn't reflect reality (new articles aren't average)
- Biases evaluation

**Option 2: Decay Training Popularity**
```python
# Reduce training popularity to simulate age
training_popularity *= 0.5 # 50% decay
```
**Why rejected:**
- Arbitrary decay rate
- Penalizes still-relevant training articles
- Doesn't solve testing = 0 problem

**Option 3: Accept Zero (Current)**
```python
# Testing articles get 0.0 popularity
testing_popularity = 0.0
```
**Why chosen:**
- Represents a realistic and unbiased evaluation setup (no artificial signals)
- Tests robustness of other features
- Matches real-world deployment scenario

---

### 9. Impact Assessment

#### Overall Impact

**Weights:**
- **Popularity**: 15% of total score
- **Other features**: 85% still contribute

**Coverage:**
- **Training articles**: 99.84% have popularity boost
- **Testing articles**: 0% have popularity boost (handicap)

#### Design Outcomes

**For Training Articles:**
- Popular articles ranked higher (viral content surfaces)
- Niche articles still visible if category/language matches
- Popularity breaks ties (2 similar articles to recommend popular one)

**For Testing Articles:**
- All start equal (no popularity advantage)
- Rely entirely on metadata matching:
  - Category similarity (50%)
  - Language match (15%)
  - News type (10%)
  - Geographic relevance (10%)
- Total possible score: **0.85** (vs 1.0 for training with max popularity)

#### When Popularity Matters Most

**Scenario 1: Two articles, similar categories, same language**
```python
Article A: category_sim=0.60, lang=1.0, popularity=0.85 (training)
Article B: category_sim=0.60, lang=1.0, popularity=0.10 (training)

Score A = 0.60*0.50 + 1.0*0.15 + 0.85*0.15 + ... = 0.73
Score B = 0.60*0.50 + 1.0*0.15 + 0.10*0.15 + ... = 0.62

Result: Article A wins (popularity broke the tie)
```

**Scenario 2: Testing vs Training (similar content)**
```python
Article C: category_sim=0.60, lang=1.0, popularity=0.0 (testing)
Article D: category_sim=0.55, lang=1.0, popularity=0.85 (training)

Score C = 0.60*0.50 + 1.0*0.15 + 0.0*0.15 + ... = 0.60
Score D = 0.55*0.50 + 1.0*0.15 + 0.85*0.15 + ... = 0.70

Result: Training article wins (popularity + category beats pure category)
```

**Scenario 3: Strong category match overcomes popularity**
```python
Article E: category_sim=0.90, lang=1.0, popularity=0.0 (testing)
Article F: category_sim=0.50, lang=1.0, popularity=1.0 (training)

Score E = 0.90*0.50 + 1.0*0.15 + 0.0*0.15 + ... = 0.75
Score F = 0.50*0.50 + 1.0*0.15 + 1.0*0.15 + ... = 0.70

Result: Testing article wins (excellent category match > popularity)
```

**Key Insight:**
- Popularity is **influential but not dominant** (15% weight)
- Testing articles can still rank high with strong content matching
- System is **robust** to cold start (doesn't collapse without popularity)

---

### 10. Key Decisions Rationale

#### Decision 1: log₁₊ Transformation (vs Linear)
- **Chosen**: `log₁₊(x) = log(1 + x)`
- **Rejected**: Linear scaling or raw counts
- **Why**:
  - Dampens outliers (viral article with 10K users doesn't dominate)
  - Captures diminishing returns (1000 to 1001 users less impactful than 10 to 11)
  - Stabilizes distribution (reduces variance)
  - Standard practice in information retrieval

**Example Impact:**
```python
# Linear scaling
Article A: 5000 users → score = 5000
Article B: 100 users → score = 100
Ratio: 50:1 (Article A dominates)

# Log scaling
Article A: log₁₊(5000) = 8.52 → normalized = 0.95
Article B: log₁₊(100) = 4.62 → normalized = 0.52
Ratio: 1.8:1 (Article A still higher but B remains visible)
```

#### Decision 2: 70/30 Weight (Unique Users vs Total Events)
- **Chosen**: 70% unique users, 30% total events
- **Rejected**: 50/50, 80/20, or 100% unique users
- **Why**:
  - **Breadth > Depth**: More important that many people read (viral potential)
  - **Quality signal**: Unique users = diverse appeal
  - **Spam resistance**: Prevents gaming via repeated views from few users
  - **Still rewards engagement**: 30% weight captures bookmarks/shares

**Example Impact:**
```python
# Article X: 1000 unique users, 1500 total events (1.5 events/user)
score_X = log₁₊(1000)*0.7 + log₁₊(1500)*0.3
        = 6.91*0.7 + 7.31*0.3
        = 4.84 + 2.19 = 7.03

# Article Y: 500 unique users, 3000 total events (6 events/user)
score_Y = log₁₊(500)*0.7 + log₁₊(3000)*0.3
        = 6.21*0.7 + 8.01*0.3
        = 4.35 + 2.40 = 6.75

Result: Article X wins (broader appeal > deep engagement)
```

#### Decision 3: Accept Zero for Testing (vs Proxy Values)
- **Chosen**: Testing articles = 0.0 popularity
- **Rejected**: Use training mean (0.470) or category-based proxy
- **Why**:
  - **Honest evaluation**: No artificial signals
  - **Real-world simulation**: New articles have no history
  - **Tests robustness**: Validates other features work independently
  - **No bias**: All testing articles start equal
  - **Simple**: No complex imputation logic

**Impact:**
- Testing recommendations rely on content matching (85% of score)
- Training recommendations get popularity boost (15% extra)
- This is **expected and acceptable** in production systems

---

### 11. Architecture Comparison

| Feature | Category Similarity | Language Match | Popularity Score |
|----------------------|-----------------------------|-----------------------------|----------------------------------|
| **Weight** | 50% | 15% | 15% |
| **Source** | TF-IDF from events | Inferred from events | Aggregated from events |
| **Coverage** | 100% users, 99.63% articles | 100% users, 98.77% articles | N/A users, 99.84% train, 0% test |
| **User-Specific** | Yes (personalized) | Yes (personalized) | No (global) |
| **Storage** | Sparse 2D matrices | Dense 2D matrix | 1D dict to vector |
| **Computation** | On-the-fly (per batch) | Precomputed (once) | Precomputed (once) |
| **Memory** | ~200 KB | ~64 MB | ~500 KB |
| **Implementation** | `cosine_similarity()` | Broadcasting `==` | Dict lookup + broadcast |
| **Missing Handling** | Natural (0 similarity) | Inference from behavior | Default 0.0 |
| **Cold Start** | No issue (use categories) | No issue (infer language) | Testing articles = 0 |

---

## 3. Language Match (15% Weight)

### 1. Overview
- **Weight**: 15% of total content-based score
- **User Source**: Inferred from reading history in `events.csv`
- **Article Source**: `newsLanguage` column from content CSV files
- **Method**: Binary matching (1.0 if languages match, 0.0 if mismatch)
- **Implementation**: Precomputed matrix for fast lookup

---

### 2. Source Data

#### User Language Preferences
- **File**: `assignment/data/processed/events.csv` + content files
- **Explicit Preferences**: `user_profiles.csv['language_preference']` (7.87% coverage)
- **Inferred Preferences**: Derived from user's reading history (99.99% coverage)
- **Users in System**: **8,689** (users with category profiles)

**Language Inference Strategy:**
```python
# For each user, analyze their reading history
# Extract newsLanguage from all articles they've read
# Use mode (most frequent language) as inferred preference
#
# Coverage:
# - 684 users (7.87%): Explicit preference in devices.csv
# - 8,688 users (99.99%): Inferred from reading history
# - 1 user (0.01%): Default to 'english' (no language data available)
```

#### Article Language Distribution
- **Files**: `training_content.csv` + `testing_content.csv`
- **Total Articles**: **9,140**
- **Key Column**: `newsLanguage` (language code)

**Language Statistics:**

| Language | Count | Percentage | Coverage |
|--------------|-------|------------|----------------|
| **english** | 6,575 | 71.94% | Primary |
| **hindi** | 2,216 | 24.25% | Secondary |
| **Missing** | 112 | 1.23% | Fill with mode |
| **telugu** | 94 | 1.03% | Regional |
| **kannada** | 76 | 0.83% | Regional |
| **gujarati** | 64 | 0.70% | Regional |
| **ANI** | 1 | 0.01% | Error/noise |
| **Twitter** | 1 | 0.01% | Error/noise |
| **भाषा** | 1 | 0.01% | Error/noise |

---

### 3. Language Matrix Creation

#### Precomputation Strategy
Unlike category similarity (computed on-the-fly), language matching uses **precomputed full matrix** for efficiency:

```python
# Shape: (8,689 users × 970 test articles)
# Stored in memory: ~64 MB (well within practical production limits)
# Computation: Fast (simple equality check)

def _precompute_features(self):
    # Extract user languages (from inferred preferences)
    user_langs = self.user_profiles['final_language'].values
    # Shape: (8689,)

    # Extract article languages (filled missing with 'english')
    article_langs = self.test_articles['newsLanguage'].fillna('english').values
    # Shape: (970,)

    # Broadcasting: (8689, 1) == (1, 970) → (8689, 970)
    self.lang_matrix = (
        user_langs[:, np.newaxis] == article_langs[np.newaxis, :]
    ).astype(float)
    # Values: 1.0 (match) or 0.0 (mismatch)
```

**Why Precompute?**
- **Computationally inexpensive**: Simple equality check (~50ms for full matrix)
- **Small memory**: 64 MB for (8,689 × 970) dense matrix
- **No repeated work**: Computed once, used across all batches
- **Fast lookup**: Direct array slicing during recommendation

**vs Category Similarity (On-the-Fly):**
- Category: Expensive cosine similarity, sparse to dense (save memory)
- Language: Cheap equality, small memory footprint (precompute)

---

### 4. Missing Data Handling

#### User Language (92% Missing in Raw Data)

**Problem:**
- `user_profiles.csv['language_preference']`: Only **7.87% coverage** (684/8,689 users)
- Cannot default all 92% to 'english' - creates bias

**Solution: Behavioral Inference**

```python
def infer_user_language(user_id, events_df, article_features_df):
    """
    Infer user language from reading history

    Returns:
        Most-read language (mode) from user's article consumption
    """
    # Get articles user has read
    user_articles = events_df[events_df['deviceId'] == user_id]['hashId']

    # Get languages of those articles
    article_langs = article_features_df[
        article_features_df['hashid'].isin(user_articles)
    ]['newsLanguage']

    # Return most common language
    if len(article_langs.mode()) > 0 and pd.notna(article_langs.mode()[0]):
        return article_langs.mode()[0]
    else:
        return 'english' # Ultimate fallback
```

**Coverage After Inference:**

| Strategy | Users | Percentage | Method |
|---------------------------|-------|------------|-----------------------------------------------|
| **Explicit Preference** | 684 | 7.87% | From devices.csv |
| **Inferred from History** | 8,004 | 92.12% | Mode of read articles |
| **Default Fill** | 1 | 0.01% | User only read articles with missing language |
| **Total Coverage** | 8,689 | 100% | Complete |

**Note on the 1 default user:**
- This user **has reading history** (that's why they're in the system)
- But **all articles they read** have `newsLanguage = NaN` (missing data)
- Cannot compute mode from empty values to Default to 'english'
- This is an extreme edge case (1 out of 8,689 users = 0.01%)

**Why This Works:**
- All 8,689 users have reading history (100% coverage - from category profiles)
- Reading behavior reveals true preference (Hindi readers to Hindi articles)
- Avoids bias (no arbitrary English default for 92%)
- Behavioral truth: If user reads Hindi, recommend Hindi

---

#### Article Language (1.23% Missing)

**Training Articles:**
- Total: 8,170
- Missing: 109 (1.33%)
- **Fill with**: 'english' (mode = 71.94%)

**Testing Articles:**
- Total: 970
- Missing: 3 (0.31%)
- **Fill with**: 'english' (mode = 71.94%)

**Justification:**
- Only 1.23% affected (minimal impact)
- 'english' is **71.94%** of all articles (statistically valid mode)
- Not creating user bias (user side uses inference)

```python
# Article language preprocessing
article_langs = test_articles['newsLanguage'].fillna('english').values
```

---

### 5. Language Matching Implementation

#### Storage Format

**Input Arrays (1D):**
```python
# From DataFrames
user_profiles['final_language']: # (8689,) ['english', 'hindi', 'english', ...]
test_articles['newsLanguage']: # (970,) ['english', 'hindi', 'telugu', ...]
```

**Precomputed Matrix (2D):**
```python
self.lang_matrix # Shape: (8689, 970)
# Memory: 8689 × 970 × 8 bytes = 67 MB
# Type: Dense NumPy array (float64)
# Values: 1.0 (match) or 0.0 (mismatch)
```

#### Usage in Recommendation

```python
def recommend_all(self, top_k=50, batch_size=3000):
    """Generate recommendations for all users (batch processing)"""

    for batch_start in range(0, total_users, batch_size):
        user_indices = list(range(batch_start, batch_end))

        # 1. Category similarity (50%) - computed on-the-fly
        category_sim = cosine_similarity(user_tfidf_batch, self.article_tfidf)

        # 2. Language scores (15%) - slice precomputed matrix
        lang_scores = self.lang_matrix[user_indices, :]
        # Shape: (3000, 970) for batch of 3000 users

        # 3. Other components...

        # Weighted combination
        content_scores = (
            category_sim * 0.50 +
            lang_scores * 0.15 + # ← Language contribution
            pop_scores_matrix * 0.15 +
            type_scores * 0.10 +
            geo_scores * 0.10
        )
```

**Example Scoring:**
```python
# User prefers Hindi, Article is Hindi
lang_score = 1.0 × 0.15 = 0.15 (full boost)

# User prefers Hindi, Article is English
lang_score = 0.0 × 0.15 = 0.00 (no boost)

# For user with 50% category match, 1.0 language match:
content_score = 0.50*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.5*0.10
              = 0.25 + 0.15 + 0.12 + 0.10 + 0.05
              = 0.67 (strong recommendation)
```

---

### 6. User Count Explanation

**Discrepancy: 8,977 vs 8,689**

From README Data Architecture:
```
events.csv → Unique users: 8,977
user_category_tfidf.pkl → 8,689 users
```

**Why 288 users missing?**

| Category | Count | Explanation |
|-----------------------------------------------------|-------|-----------------------------|
| **Total users in events.csv** | 8,977 | All users who read articles |
| **Users who read articles WITH categories** | 8,689 | Can build category profiles |
| **Users who ONLY read articles WITHOUT categories** | 288 | No category data to excluded |

**Validation:**
```python
# These 288 users only read articles with missing 'categories' column
# Cannot compute TF-IDF profiles (no category data)
# Correctly excluded from recommendation system
#
# Impact: No loss - these users have insufficient data for content-based recommendations
```

**For Language Matching:**
- System uses 8,689 users (same as category profiles)
- 8,688 users (99.99%) get language inferred from reading history
- 1 user (0.01%) gets default 'english'

---

### 7. Architecture Comparison

| Feature | Category Similarity | Language Match |
|----------------------|-------------------------------|--------------------------|
| **Weight** | 50% | 15% |
| **User Source** | TF-IDF from events.csv | Inferred from events.csv |
| **Article Source** | TF-IDF from categories | newsLanguage column |
| **User Coverage** | 100% (8,689/8,689) | 100% (8,689/8,689) |
| **Article Coverage** | 99.63% | 98.77% |
| **Storage** | Sparse (2D: 8689×31 + 970×31) | Dense 1D arrays to Matrix |
| **Computation** | On-the-fly (per batch) | Precomputed (once) |
| **Memory** | ~200 KB (sparse) | ~64 MB (dense matrix) |
| **Implementation** | `cosine_similarity()` | Broadcasting `==` |
| **Missing Handling** | Natural (0 similarity) | Inference from behavior |

---

### 8. Impact Assessment

**Overall Impact:**
- **15% of total score** - significant but secondary to categories
- **99.99% user coverage** - excellent (only 1 user defaulted)
- **98.77% article coverage** - excellent (112 articles filled)

**Design Outcomes:**
- **Hindi users** to Hindi articles ranked higher (0.15 boost)
- **English users** to English articles ranked higher (0.15 boost)
- **Regional users** to Regional language articles boosted
- **Diverse content** to Still visible (other 85% of score contributes)

**When Language Match Matters Most:**
- Two articles with similar categories (0.50 × 50% = 0.25 each)
- Language match breaks the tie: +0.15 vs +0.00
- Hindi user chooses Hindi article over English article

**Robustness:**
- Behavioral inference (not arbitrary defaults)
- Only 0.01% users need default
- Validates to business logic (readers get their language)

---

### 9. Key Decisions Rationale

#### Decision 1: Infer from Reading History (vs Default 'english')
- **Chosen**: Infer from user's most-read language
- **Rejected**: Default all 92% to 'english'
- **Why**: Avoids bias, uses actual behavior, 99.99% coverage

#### Decision 2: Precompute Matrix (vs Compute On-the-Fly)
- **Chosen**: Precompute (8689 × 970) matrix once
- **Rejected**: Compute per batch
- **Why**: Computationally inexpensive (50ms), small memory (64 MB), no repeated work

#### Decision 3: Fill Articles with 'english' (vs Leave Missing)
- **Chosen**: Fill 112 articles (1.23%) with 'english'
- **Rejected**: Leave as NaN or use neutral 0.5
- **Why**: Minimal impact (1.23%), mode is statistically valid (71.94%), simple implementation

---

## 4. Type Match (10% Weight)

### 1. Overview
- **Weight**: 10% of total content-based score
- **User Source**: `user_profiles.csv['preferred_newsType']` (inferred from reading history)
- **Article Source**: `article_features.csv['newsType']` column
- **Method**: Binary matching (1.0 if types match, 0.0 if mismatch)
- **Coverage**: Users 100%, Training 95.75%, Testing 99.90%
- **Implementation**: Precomputed matrix for fast lookup

---

### 2. Source Data

#### User Type Preferences
- **File**: `assignment/data/processed/user_profiles.csv`
- **Column**: `preferred_newsType`
- **Coverage**: **100% (8,689/8,689 users)**
- **Method**: Mode (most-read type) from user's reading history in `events.csv`

**User Type Distribution:**

| Type | Users | Percentage | Notes |
|-----------------|----------|------------|------------------------------------|
| **NEWS** | ~8,500+ | ~98%+ | Vast majority prefer standard news |
| **VIDEO_NEWS** | ~100-200 | ~1-2% | Video content consumers |
| **Other types** | ~few | <1% | Edge cases |

**Inference Strategy:**
```python
# For each user, analyze reading history
user_articles = events[events['deviceId'] == user]['hashId']
article_types = article_features[
    article_features['hashid'].isin(user_articles)
]['newsType']

# Mode = most-read type
preferred_type = article_types.mode()[0]
# Result: 100% coverage (all users have reading history)
```

#### Article Type Distribution
- **Files**: `training_content.csv` + `testing_content.csv`
- **Total Articles**: **9,140**
- **Key Column**: `newsType`

**Training Articles (8,170 total):**

| Status | Count | Percentage | Notes |
|-------------------|-------|------------|------------------|
| **Has newsType** | 7,823 | 95.75% | Valid type data |
| **Missing (NaN)** | 347 | 4.25% | Fill with 'NEWS' |

**Testing Articles (970 total):**

| Status | Count | Percentage | Notes |
|-------------------|-------|------------|--------------------|
| **Has newsType** | 969 | 99.90% | coverage |
| **Missing (NaN)** | 1 | 0.10% | Fill with 'NEWS' |

**Overall Statistics:**
- **Total missing**: 348 / 9,140 = **3.81%**
- **Coverage**: 96.19% articles have type data

**Type Categories:**

| Type | Count | Percentage | Description |
|-----------------|----------|------------|------------------------|
| **NEWS** | ~8,500+ | ~93%+ | Standard text articles |
| **VIDEO_NEWS** | ~300-500 | ~3-5% | Video content |
| **Other types** | ~100-200 | ~1-2% | Special formats |
| **Missing** | 348 | 3.81% | Fill with 'NEWS' |

---

### 3. Type Matching Implementation

#### Precomputation Strategy

Similar to language matching, type uses **precomputed full matrix**:

```python
def _precompute_features(self):
    # Extract user preferred types (from inference)
    user_types = self.user_profiles['preferred_newsType'].values
    # Shape: (8689,)

    # Extract article types (filled missing with 'NEWS')
    article_types = self.test_articles['newsType'].fillna('NEWS').values
    # Shape: (970,)

    # Broadcasting: (8689, 1) == (1, 970) → (8689, 970)
    self.type_matrix = (
        user_types[:, np.newaxis] == article_types[np.newaxis, :]
    ).astype(float)
    # Values: 1.0 (match) or 0.0 (mismatch)
```

**Matrix Details:**
- **Shape**: (8,689 users × 970 test articles)
- **Memory**: ~64 MB (same as language matrix)
- **Type**: Dense NumPy array (float64)
- **Values**: Binary (1.0 or 0.0)

**Why Precompute?**
- **Computationally inexpensive**: Simple equality check (~50ms)
- **Small memory**: 64 MB acceptable overhead, well within acceptable production memory budgets
- **No repeated work**: Computed once, used across all batches
- **Fast lookup**: Direct array slicing

---

### 4. Missing Data Handling

#### User Type (0% Missing)

**Problem: Solved**
- All users (8,689) have reading history
- Can infer preferred type from articles read
- **Coverage: 100%**

**Inference Logic:**
```python
def infer_user_type(user_id, events_df, article_features_df):
    """Infer user's preferred newsType from reading history"""

    # Get articles user has read
    user_articles = events_df[events_df['deviceId'] == user_id]['hashId']

    # Get types of those articles (ignore NaN)
    article_types = article_features_df[
        article_features_df['hashid'].isin(user_articles)
    ]['newsType'].dropna() # ← Ignore missing types in articles

    # Return most common type
    if len(article_types.mode()) > 0:
        return article_types.mode()[0]
    else:
        return 'NEWS' # Ultimate fallback (if all articles have NaN type)
```

**Coverage After Inference:**

| Strategy | Users | Percentage | Method |
|---------------------------|-------|------------|----------------------------|
| **Inferred from History** | 8,689 | 100% | Mode of read article types |
| **Default Fill** | 0 | 0% | Not needed |
| **Total Coverage** | 8,689 | 100% | Complete |

---

#### Article Type (3.81% Missing)

**Training Articles:**
- Total: 8,170
- Missing: 347 (4.25%)
- **Fill with**: 'NEWS' (mode ≈ 93%)

**Testing Articles:**
- Total: 970
- Missing: 1 (0.10%)
- **Fill with**: 'NEWS' (mode ≈ 93%)

**Justification:**
- Only 3.81% affected (minimal impact)
- 'NEWS' is dominant type (~93% of articles)
- Statistically valid mode
- Simple binary matching (no complex logic)

```python
# Article type preprocessing
article_types = test_articles['newsType'].fillna('NEWS').values
```

**Critical Order (Same as Language):**

**Step 1: Infer User Type (Ignore NaN)**
```python
# User reads articles, some have NaN type
article_types = user_articles['newsType'].dropna() # Ignore NaN
user_preferred_type = article_types.mode()[0] # Compute on clean data
```

**Step 2: Fill Article Types for Matching**
```python
# NOW fill articles for the matching matrix
article_types_filled = articles['newsType'].fillna('NEWS')
```

**Why This Order Matters:**
- Prevents bias: 348 NaN articles shouldn't force users to 'NEWS' preference
- Preserves true behavior: User's actual reading pattern determines preference
- Filling only for matching: NaN to 'NEWS' only affects final comparison

---

### 5. Type Distribution

#### User Preferences

```python
# From user_profiles.csv (8,689 users)
preferred_newsType distribution:
  NEWS: ~8,500 users (~98%)
  VIDEO_NEWS: ~150 users (~1.7%)
  Others: ~39 users (~0.3%)
```

**Key Insight:**
- **Highly skewed**: 98% prefer standard NEWS articles
- **Video niche**: Only ~2% prefer VIDEO_NEWS
- **Matching crucial**: Video users MUST get video content (10% boost)

#### Article Distribution

```python
# From article_features.csv (9,140 articles)
newsType distribution:
  NEWS: ~8,500 articles (~93%)
  VIDEO_NEWS: ~400 articles (~4%)
  Others: ~240 articles (~3%)
  Missing: 348 articles (3.81%)
```

**Observations:**
- **Supply-demand mismatch**: 4% video articles vs 2% video users
- **Adequate variety**: Enough content for each user segment
- **NEWS dominance**: Vast majority are standard text articles

---

### 6. Implementation Details

#### Storage Format

**Input Arrays (1D):**
```python
# From DataFrames
user_profiles['preferred_newsType']: # (8689,) ['NEWS', 'VIDEO_NEWS', 'NEWS', ...]
test_articles['newsType']: # (970,) ['NEWS', 'VIDEO_NEWS', 'NEWS', ...]
```

**Precomputed Matrix (2D):**
```python
self.type_matrix # Shape: (8689, 970)
# Memory: 8689 × 970 × 8 bytes = 67 MB
# Type: Dense NumPy array (float64)
# Values: 1.0 (match) or 0.0 (mismatch)
```

#### Usage in Recommendation

```python
def recommend_all(self, top_k=50, batch_size=3000):
    """Generate recommendations for all users (batch processing)"""

    for batch_start in range(0, total_users, batch_size):
        user_indices = list(range(batch_start, batch_end))

        # 1. Category similarity (50%)
        category_sim = cosine_similarity(user_tfidf_batch, self.article_tfidf)

        # 2. Language scores (15%)
        lang_scores = self.lang_matrix[user_indices, :]

        # 3. Popularity scores (15%)
        pop_scores_matrix = np.broadcast_to(self.pop_vector, (n_batch_users, 970))

        # 4. Type scores (10%) - slice precomputed matrix
        type_scores = self.type_matrix[user_indices, :]
        # Shape: (3000, 970) for batch of 3000 users

        # 5. Other components...

        # Weighted combination
        content_scores = (
            category_sim * 0.50 +
            lang_scores * 0.15 +
            pop_scores_matrix * 0.15 +
            type_scores * 0.10 + # ← Type contribution
            geo_scores * 0.10
        )
```

**Example Scoring:**
```python
# User prefers NEWS, Article is NEWS
type_score = 1.0 × 0.10 = 0.10 (full boost)

# User prefers NEWS, Article is VIDEO_NEWS
type_score = 0.0 × 0.10 = 0.00 (no boost)

# For user with 60% category match, perfect language, NEWS match:
content_score = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.5*0.10
              = 0.30 + 0.15 + 0.12 + 0.10 + 0.05
              = 0.72 (strong recommendation)
```

---

### 7. Coverage Analysis

#### User Coverage: 100%

| Status | Users | Percentage | Method |
|-------------------------------|-------|------------|----------------------|
| **Has preference (inferred)** | 8,689 | 100% | From reading history |
| **Missing** | 0 | 0% | None |
| **Total** | 8,689 | 100% | Complete |

**Why Coverage?**
- All users in system have reading history (by definition)
- Can always compute mode from articles read
- Even if articles have NaN, we use `.dropna()` first

#### Article Coverage

**Training Articles (8,170 total):**

| Status | Count | Percentage | Type Value |
|-----------------------|-------|------------|------------------|
| **Has newsType** | 7,823 | 95.75% | Original data |
| **Missing (filled)** | 347 | 4.25% | 'NEWS' (default) |
| **Total** | 8,170 | 100% | Complete |

**Testing Articles (970 total):**

| Status | Count | Percentage | Type Value |
|----------------------|-------|------------|------------------|
| **Has newsType** | 969 | 99.90% | Original data |
| **Missing (filled)** | 1 | 0.10% | 'NEWS' (default) |
| **Total** | 970 | 100% | Complete |

**Overall:**
- **96.19% articles** have original type data
- **3.81% articles** filled with 'NEWS'
- **0% users** need default (all inferred successfully)

---

### 8. Impact Assessment

#### Overall Impact

**Weights:**
- **Type Match**: 10% of total score
- **Other features**: 90% still contribute

**Coverage:**
- **Users**: 100% have preferences
- **Articles**: 96.19% have original data

#### Design Outcomes

**For NEWS Users (~98% of users):**
- NEWS articles get +0.10 boost
- VIDEO_NEWS articles get 0.00 (filtered out)
- Ensures text-focused users see text content

**For VIDEO_NEWS Users (~2% of users):**
- VIDEO_NEWS articles get +0.10 boost
- NEWS articles get 0.00 (filtered out)
- Critical for niche audience satisfaction

**For Mixed Content:**
- Type match breaks ties between similar articles
- Ensures content format preference is respected

#### When Type Match Matters Most

**Scenario 1: Video user, similar category articles**
```python
Article A: category_sim=0.60, lang=1.0, pop=0.8, type=VIDEO_NEWS
Article B: category_sim=0.60, lang=1.0, pop=0.8, type=NEWS

User prefers VIDEO_NEWS:
Score A = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.5*0.10
        = 0.30 + 0.15 + 0.12 + 0.10 + 0.05 = 0.72
Score B = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 0.0*0.10 + 0.5*0.10
        = 0.30 + 0.15 + 0.12 + 0.00 + 0.05 = 0.62

Result: Video article wins (type match broke the tie)
```

**Scenario 2: Type overcomes weak category match**
```python
Article C: category_sim=0.40, type=VIDEO_NEWS (matches user)
Article D: category_sim=0.50, type=NEWS (doesn't match user)

User prefers VIDEO_NEWS:
Score C = 0.40*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.5*0.10
        = 0.20 + 0.15 + 0.12 + 0.10 + 0.05 = 0.62
Score D = 0.50*0.50 + 1.0*0.15 + 0.8*0.15 + 0.0*0.10 + 0.5*0.10
        = 0.25 + 0.15 + 0.12 + 0.00 + 0.05 = 0.57

Result: Weaker category match wins due to type preference
```

**Key Insight:**
- Type is **influential for niche users** (video consumers)
- For majority (NEWS users), acts as **filter** (eliminates video content)
- 10% weight is appropriate (not dominant but meaningful)

---

### 9. Key Decisions Rationale

#### Decision 1: Infer from Reading History (vs Use Explicit Preference)

- **Chosen**: Infer from user's most-read type
- **Rejected**: Use explicit preference from `devices.csv` (if available)
- **Why**:
  - Behavioral truth > stated preference
  - 100% coverage via inference (no missing data)
  - Reflects actual consumption patterns
  - Users reveal true preference through actions

#### Decision 2: Binary Matching (vs Graduated Scoring)

- **Chosen**: 1.0 (match) or 0.0 (mismatch)
- **Rejected**: Graduated scores (e.g., NEWS to VIDEO_NEWS = 0.5)
- **Why**:
  - Clear user intent: Video users want video, not "kind of video"
  - Simple implementation (equality check)
  - Types are categorical (not ordinal - no natural ordering)
  - Prevents cross-contamination (text users don't want video)

**Example of Why Binary Works:**
```python
# Binary (correct)
User prefers VIDEO_NEWS:
  VIDEO_NEWS article → 1.0 NEWS article → 0.0 # Graduated (wrong)
User prefers VIDEO_NEWS:
  VIDEO_NEWS article → 1.0 NEWS article → 0.5 (user doesn't want text at all!)
```

#### Decision 3: Fill Missing with 'NEWS' (vs Leave NaN)

- **Chosen**: Fill 348 articles (3.81%) with 'NEWS'
- **Rejected**: Leave as NaN or use neutral 0.5
- **Why**:
  - 'NEWS' is mode (93% of articles)
  - Statistically valid assumption
  - Minimal impact (3.81% affected)
  - Simple implementation
  - Only for matching (user inference uses `.dropna()`)

#### Decision 4: Precompute Matrix (vs Compute On-the-Fly)

- **Chosen**: Precompute (8689 × 970) matrix once
- **Rejected**: Compute per batch
- **Why**:
  - Computationally inexpensive (equality check ~50ms)
  - Small memory (64 MB, well within practical production limits)
  - No repeated work (used across all batches)
  - Consistent with language matching strategy

---

### 10. Architecture Comparison

| Feature | Category Similarity | Language Match | Type Match |
|----------------------|------------------------|-------------------------|-------------------------|
| **Weight** | 50% | 15% | 10% |
| **User Source** | TF-IDF from events | Inferred from events | Inferred from events |
| **Article Source** | TF-IDF from categories | newsLanguage column | newsType column |
| **User Coverage** | 100% (8,689/8,689) | 100% (8,689/8,689) | 100% (8,689/8,689) |
| **Article Coverage** | 99.63% | 98.77% | 96.19% |
| **Storage** | Sparse 2D matrices | Dense 2D matrix | Dense 2D matrix |
| **Computation** | On-the-fly (per batch) | Precomputed (once) | Precomputed (once) |
| **Memory** | ~200 KB (sparse) | ~64 MB (dense) | ~64 MB (dense) |
| **Implementation** | `cosine_similarity()` | Broadcasting `==` | Broadcasting `==` |
| **Missing Handling** | Natural (0 similarity) | Inference from behavior | Inference from behavior |
| **Matching Type** | Continuous [0, 1] | Binary (1.0 or 0.0) | Binary (1.0 or 0.0) |

---

### 11. User Preference Validation

#### Why 100% Coverage is Guaranteed

**Prerequisite for Being in System:**
```python
# User must meet this condition to be in user_profiles.csv:
1. User appears in events.csv (has reading history)
2. User has read articles WITH categories (can build TF-IDF profile)

# This means:
- All 8,689 users have reading history (by definition)
- All have read multiple articles
- Can always compute mode(newsType) from those articles
```

**Edge Case Handling:**
```python
# Even if all articles user read have NaN type:
article_types = user_articles['newsType'].dropna()

if len(article_types) > 0:
    preferred_type = article_types.mode()[0]
else:
    preferred_type = 'NEWS' # Ultimate fallback
    # This case: 0 users (all users read at least some articles with type data)
```

**Result:**
- **0 users** need default 'NEWS'
- **8,689 users** (100%) successfully inferred
- **Robust**: Even extreme edge cases are handled

---

## 5. Geographic Score (10% Weight)

**Purpose**: Boost local content relevance while handling sparse location data

**Key Characteristics:**
- **Weight**: 10% of total content score
- **Method**: Rule-based scoring with inferred article locations
- **User Coverage**: 7.9% (710/8,977 users have city data)
- **Article Coverage**: 100% (all articles tagged via inference)
- **Strategy**: Soft boosting (neutral for unknown, boost for local match)

---

### 1. Problem Statement

#### Challenge: Sparse User Location Data

**User Location Quality (from `devices.csv`):**

| Field | Coverage | Usability |
|--------------------------------|---------------------|-----------------|
| `district` | 0.2% (21 users) | Too sparse |
| `lastknownsubadminarea` (city) | 91.3% (9,492/10,400) | Best available |

**But Critical Issue:**
- Only **710 users (7.9%)** appear in BOTH `devices.csv` AND `events.csv`
- 92% of active users have no location data
- Cannot rely on explicit location tags

#### Challenge: Sparse Article Location Data

**Article Location Quality:**

| Source | Training Coverage | Testing Coverage | Usability |
|--------------------------------|-------------------|------------------|-------------------|
| `newsDistrict` | 0.40% | 32.27% | Too sparse |
| Inferred from reader geography | 100% | 100% | Behavioral signal |

**Solution**: Infer article locations from reader geography

---

### 2. Source Data

#### User Location: `devices.csv`

**Field Used**: `lastknownsubadminarea` (city name)
- **Format**: "Mumbai", "Delhi", "Bengaluru", etc.
- **Coverage**: 9,492 users (91.3% of devices table)
- **But**: Only 710 users (7.9%) overlap with `events.csv`

**Why Not Use Other Fields:**
- `district`: Only 21 values (0.2% coverage) - unusable
- `state`: Too broad (multiple cities per state)
- `locality`: Too specific (varies within city)

#### Article Location: Inferred from Reader Geography

**Method**: Behavioral Inference from `events.csv`
```python
# For each article, analyze which cities read it
city_distribution = events.merge(devices).groupby(['hashId', 'user_city']).size()

# Rule: If ≥50% of reads from one city → tag article with that city
if max_city_percentage >= 0.50:
    article_location = dominant_city
else:
    article_location = 'NATIONAL' # Broad appeal
```

**Coverage:**
- Training articles in events: 100% tagged (via inference)
- Testing articles (cold start): 100% tagged as 'NATIONAL'
- Result: ~10% city-specific, ~90% NATIONAL
- Inference data: 2.94% of events (104,069/3,544,161) come from users with city data

---

### 3. Article Location Inference Process

#### Step 1: Extract Users with City Data

```python
# Get users who have city information
user_cities = devices[['deviceid', 'lastknownsubadminarea']].dropna()

# Result: 9,492 users with city data
# But only 684 appear in events (7.9% coverage)
```

#### Step 2: Merge Events with User Cities

```python
# Join events with user cities (inner join)
events_with_city = events.merge(user_cities, on='deviceId', how='inner')

# Result: ~2-3% of events have city data
# But enough to infer article geography patterns
```

#### Step 3: Calculate City Distribution per Article

```python
# For each article, count reads from each city
article_city_counts = events_with_city.groupby(['hashId', 'user_city']).size()

# Calculate percentage
article_city_counts['percentage'] = reads / total_reads_per_article
```

**Example:**
```
Article A:
  Mumbai: 120 reads (65%) → Tag as 'Mumbai'
  Delhi: 40 reads (22%)
  Others: 25 reads (13%)

Article B:
  Mumbai: 50 reads (35%)
  Delhi: 45 reads (32%) → Tag as 'NATIONAL'
  Others: 48 reads (33%)
```

#### Step 4: Apply Inference Rule (50% Threshold)

```python
if max_city_percentage >= 0.50:
    article_location = max_city
else:
    article_location = 'NATIONAL'
```

**Why 50% Threshold:**
- Conservative: Requires clear majority from one city
- Prevents false positives: Article with 35% Mumbai, 30% Delhi = NATIONAL (correct)
- Statistically sound: >50% is strong signal of local relevance

#### Step 5: Handle Cold Start (Testing Articles)

```python
# Testing articles have NO interaction history
for testing_article in testing_content:
    if article not in article_locations:
        article_locations[article] = 'NATIONAL' # Safe default
```

---

### 4. Coverage Analysis

#### User Coverage: 7.9% (Very Sparse!)

| Status | Count | Percentage | Source |
|------------------------------|-------|------------|---------------------|
| **Has city data in devices** | 9,492 | 91.3% | `devices.csv` |
| **Has city + in events** | 710 | 7.9% | `devices ∩ events` |
| **Missing city (UNKNOWN)** | 8,267 | 92.1% | No overlap |
| **Total active users** | 8,977 | 100% | All users in system |

**Critical Insight:**
- Only **710 users** (7.9%) have both location AND interaction history
- These 710 users provide signals for article location inference (2.94% of all events)
- Remaining 8,267 users get neutral scoring (0.7 for all articles)

#### Article Coverage: 100% (via Inference)

**Training Articles (8,170 total):**

| Location Type | Count | Percentage | Inference Source |
|-------------------|--------|------------|--------------------------|
| **City-specific** | ~800 | ~10% | ≥50% reads from one city |
| **NATIONAL** | ~7,370 | ~90% | Mixed readership |
| **Total** | 8,170 | 100% | Complete |

**Testing Articles (970 total):**

| Location Type | Count | Percentage | Inference Source |
|---------------|-------|------------|-------------------------------------|
| **NATIONAL** | 970 | 100% | No interaction history (cold start) |
| **Total** | 970 | 100% | Complete |

**Why This Works:**
- Behavioral signal more reliable than explicit tags (0.40% coverage)
- NATIONAL default **aligns with observed distribution** for 90% of articles (news has broad appeal)
- Cold start articles safely default to NATIONAL

---

### 5. Geographic Scoring Logic

#### Scoring Rules (Soft Boosting Approach)

```python
# For each user-article pair:
if user_city == article_city:
    geo_score = 1.0 # Perfect local match
elif user_city == 'UNKNOWN' or article_city == 'NATIONAL':
    geo_score = 0.7 # Neutral (not penalty)
else:
    geo_score = 0.5 # Different cities (mild penalty)
```

**Score Meanings:**

| Score | Scenario | Rationale |
|---------|----------------------------------------|---------------------------------------|
| **1.0** | User from Mumbai + Article from Mumbai | Perfect local relevance |
| **0.7** | User unknown + Any article | Neutral - no penalty for missing data |
| **0.7** | Any user + NATIONAL article | Neutral - broad appeal content |
| **0.5** | User from Mumbai + Article from Delhi | Different region, reduced relevance |

#### Why Soft Boosting (Not Hard Filtering)?

**Problem with Hard Filtering:**
```python
# If we used 1.0 or 0.0 only:
geo_score = 1.0 if match else 0.0

# Impact for 92% users with unknown location:
- All articles get 0.0 → Geographic component useless
- Penalizes majority for missing data → Unfair
```

**Solution: Soft Boosting**
```python
# Neutral default (0.7):
- Unknown users get 0.7 (not penalized)
- NATIONAL articles get 0.7 (appropriate for broad content)
- Local matches get 1.0 (boosted when we have signal)
- Cross-city gets 0.5 (mild penalty for geographic mismatch)
```

**Benefits:**
- Doesn't penalize 92% of users with missing location
- Boosts local content when we have data (710 users)
- Treats NATIONAL articles appropriately (neutral, not penalty)
- Still differentiates local vs cross-city (1.0 vs 0.5 = 2x)

---

### 6. Implementation

#### Precomputed Matrix Approach

```python
# In _precompute_features():
n_users, n_articles = 8689, 970

# Step 1: Get user cities
user_cities = user_profiles['user_city'].values # From devices.csv

# Step 2: Get article locations (inferred)
article_cities = [article_locations.get(hashid, 'NATIONAL')
                  for hashid in article_hashids]

# Step 3: Initialize with neutral score (0.7)
geo_matrix = np.full((n_users, n_articles), 0.7, dtype=float)

# Step 4: Update for users with known location
for i, user_city in enumerate(user_cities):
    if user_city != 'UNKNOWN':
        # Same city → 1.0
        same_city_mask = (article_cities == user_city)
        geo_matrix[i, same_city_mask] = 1.0

        # Different specific city → 0.5
        different_city_mask = (
            (article_cities != 'NATIONAL') &
            (article_cities != user_city)
        )
        geo_matrix[i, different_city_mask] = 0.5

        # NATIONAL articles stay 0.7 (no change)
```

**Memory**: 8,977 users × 970 articles × 8 bytes = ~70 MB (well within acceptable production memory budgets)

**Performance**:
- Precomputed once at initialization (~50ms)
- Batch lookup: O(1) per user-article pair
- No repeated computation across batches

---

### 7. Example Calculations

#### Example 1: Mumbai User + Mumbai Article

```python
user_city = 'Mumbai'
article_city = 'Mumbai' # Inferred (65% of reads from Mumbai)

geo_score = 1.0 × 0.10 = 0.10 (full boost)

# For user with 60% category match, perfect language, NEWS match:
content_score = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 1.0*0.10
              = 0.30 + 0.15 + 0.12 + 0.10 + 0.10
              = 0.77 (strong local recommendation)
```

#### Example 2: Unknown User + NATIONAL Article

```python
user_city = 'UNKNOWN' # 92% of users
article_city = 'NATIONAL' # 90% of articles

geo_score = 0.7 × 0.10 = 0.07 (neutral)

# Same scores for category/language/type:
content_score = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.7*0.10
              = 0.30 + 0.15 + 0.12 + 0.10 + 0.07
              = 0.74 (not penalized for missing data)
```

#### Example 3: Mumbai User + Delhi Article

```python
user_city = 'Mumbai'
article_city = 'Delhi' # Inferred (55% of reads from Delhi)

geo_score = 0.5 × 0.10 = 0.05 (mild penalty)

# Same scores for other components:
content_score = 0.60*0.50 + 1.0*0.15 + 0.8*0.15 + 1.0*0.10 + 0.5*0.10
              = 0.30 + 0.15 + 0.12 + 0.10 + 0.05
              = 0.72 (reduced due to geography)
```

**Impact Analysis:**
- Local match (1.0) vs Cross-city (0.5) = **0.05 score difference** (5%)
- Local match (1.0) vs Unknown/NATIONAL (0.7) = **0.03 score difference** (3%)
- With 10% weight, geography is **influential but not dominant**

---

### 8. Impact Assessment

#### Overall Impact

**Weights:**
- **Geographic Score**: 10% of total score
- **Other features**: 90% still contribute

**Coverage:**
- **Users with signal**: 7.9% (710/8,977) get boosted for local content
- **Users without signal**: 92.1% (8,267/8,977) get neutral scoring
- **Articles**: 100% tagged (10% city-specific, 90% NATIONAL)

#### Design Outcomes

**For Users with Known Location (7.9%):**
```python
# Mumbai user sees:
Mumbai article (1.0 × 0.10) = 0.10 contribution Boosted
NATIONAL article (0.7 × 0.10) = 0.07 contribution Neutral
Delhi article (0.5 × 0.10) = 0.05 contribution Penalized
```

**For Users with Unknown Location (92.1%):**
```python
# Unknown user sees:
Any article (0.7 × 0.10) = 0.07 contribution Neutral
# No penalty for missing data
# Geographic component essentially becomes constant
```

**For Articles:**
- **City-specific** (10%): Get boost for matching city users (1.0)
- **NATIONAL** (90%): Get neutral score for all users (0.7)

#### When Geographic Score Matters Most

**Scenario 1: Tie-breaking between similar articles**
```python
Article A: Mumbai local news (category=0.60, lang=1.0, pop=0.6, type=1.0, geo=1.0)
Article B: Delhi local news (category=0.60, lang=1.0, pop=0.6, type=1.0, geo=0.5)

For Mumbai user:
Score A = 0.60*0.50 + 1.0*0.15 + 0.6*0.15 + 1.0*0.10 + 1.0*0.10 = 0.74
Score B = 0.60*0.50 + 1.0*0.15 + 0.6*0.15 + 1.0*0.10 + 0.5*0.10 = 0.69

Result: Local article wins (geography broke the tie)
```

**Scenario 2: Geography overcomes weak popularity**
```python
Article C: Mumbai local event (pop=0.3, geo=1.0 for Mumbai user)
Article D: National viral story (pop=0.9, geo=0.7 for Mumbai user)

For Mumbai user:
Score C = 0.60*0.50 + 1.0*0.15 + 0.3*0.15 + 1.0*0.10 + 1.0*0.10 = 0.69
Score D = 0.60*0.50 + 1.0*0.15 + 0.9*0.15 + 1.0*0.10 + 0.7*0.10 = 0.72

Result: Viral story wins (popularity advantage too large)
```

**Key Insight:**
- Geography is **tie-breaker** for similar articles
- 10% weight is appropriate: Noticeable but not dominant
- Doesn't override strong category/popularity signals
- Enhances local relevance when we have data

---

### 9. Key Decisions Rationale

#### Decision 1: Infer Article Location from Reader Geography (vs Use Explicit Tags)

- **Chosen**: Infer from city distribution of readers (50% threshold)
- **Rejected**: Use `newsDistrict` column (only 0.40% coverage)
- **Why**:
  - Behavioral data more reliable than explicit tags
  - 100% coverage via inference vs 0.40% explicit
  - Reflects true geographic relevance (where people actually read it)
  - Handles cold start (testing articles default to NATIONAL)

#### Decision 2: Soft Boosting (vs Hard Filtering)

- **Chosen**: Scores 1.0 / 0.7 / 0.5 (neutral for unknown)
- **Rejected**: Hard filtering 1.0 / 0.0 (penalty for unknown)
- **Why**:
  - 92% users have unknown location - can't penalize majority
  - NATIONAL articles (90%) should be neutral, not penalty
  - Still boosts local content when we have signal (1.0 vs 0.5 = 2x)
  - Treats missing data fairly (neutral 0.7, not zero)

**Example of Why Soft Boosting Works:**
```python
# Hard filtering (wrong):
Unknown user + NATIONAL article → 0.0 (penalized for missing data!)

# Soft boosting (correct):
Unknown user + NATIONAL article → 0.7 (neutral, fair treatment)
```

#### Decision 3: 50% Threshold for City Tagging (vs Lower Threshold)

- **Chosen**: Require ≥50% reads from one city
- **Rejected**: Lower threshold (e.g., 40% or plurality)
- **Why**:
  - Conservative: Clear majority required
  - Prevents false positives: 35% Mumbai, 30% Delhi to NATIONAL (correct)
  - 90% NATIONAL result matches reality (most news has broad appeal)
  - Statistical soundness: >50% is strong local signal

**Example:**
```python
# With 50% threshold (correct):
Article: 45% Mumbai, 30% Delhi, 25% others
Result: 'NATIONAL' (no clear local dominance)

# With 40% threshold (risky):
Article: 45% Mumbai, 30% Delhi, 25% others
Result: 'Mumbai' (false positive - not truly local)
```

#### Decision 4: Use `lastknownsubadminarea` (vs Other Location Fields)

- **Chosen**: City-level granularity (`lastknownsubadminarea`)
- **Rejected**: `district` (0.2% coverage), `state` (too broad), `locality` (too specific)
- **Why**:
  - Best balance: Specific enough (city) yet broad enough (91.3% coverage)
  - Matches article geography granularity (Mumbai, Delhi, etc.)
  - `district` unusable (only 21 values)
  - `state` too broad (multiple cities with different news interests)

---

### 10. Architecture Comparison

| Feature | Category | Popularity | Language | Type | **Geographic** |
|----------------------|------------------|----------------------|-------------------|-------------------|-----------------------------|
| **Weight** | 50% | 15% | 15% | 10% | **10%** |
| **User Source** | TF-IDF | N/A | Inferred | Inferred | **devices.csv** |
| **Article Source** | TF-IDF | events.csv | newsLanguage | newsType | **Inferred from readers** |
| **User Coverage** | 100% | N/A | 100% | 100% | **7.9%** |
| **Article Coverage** | 99.6% | 99.8% train, 0% test | 98.8% | 96.2% | **100%** |
| **Storage** | Sparse | Dict | Dense 2D | Dense 2D | **Dense 2D** |
| **Computation** | On-the-fly | Vectorized lookup | Precomputed | Precomputed | **Precomputed** |
| **Memory** | ~200 KB | Negligible | ~64 MB | ~64 MB | **~64 MB** |
| **Implementation** | cosine_sim | Broadcasting | Broadcasting `==` | Broadcasting `==` | **Rule-based** |
| **Missing Handling** | 0 similarity | 0.0 for testing | Inference | Inference | **Neutral 0.7** |
| **Matching Type** | Continuous [0,1] | Continuous [0,1] | Binary (1/0) | Binary (1/0) | **Graduated (1.0/0.7/0.5)** |

**Key Differences:**
- **Lowest user coverage** (7.9% vs 100% for others)
- **Only feature with graduated scoring** (not binary or continuous)
- **Only feature with explicit "unknown" handling** (0.7 neutral score)
- **Soft boosting philosophy** (doesn't penalize majority with missing data)

---

### 11. Top Cities Analysis

#### Expected Top Cities (from EDA)

From EDA analysis of 710 users with city data:

| City | Users | Total Events | Avg Time Spent |
|-----------|-------|--------------|----------------|
| Mumbai | 41 | 6,061 | 7.46s |
| Delhi | 34 | 3,433 | 9.37s |
| Bengaluru | 31 | 3,340 | 11.50s |
| Kolkata | 23 | 2,798 | 10.41s |
| Noida | 21 | 4,259 | 7.88s |
| Patna | 19 | 1,107 | 18.24s |
| Lucknow | 15 | 1,482 | 9.60s |
| Gurgaon | 15 | 1,249 | 14.31s |
| Chennai | 14 | 10,677 | 5.94s |
| Hyderabad | 13 | 3,524 | 6.94s |

**Insights:**
- **Metro dominance**: Top cities are major economic hubs
- **Engagement variation**: Patna shows highest time spent (18.24s) despite lower volume
- **Chennai anomaly**: High event count but low time (5.94s) - rapid scrolling behavior

#### Expected Article Distribution

Based on inference from these 710 users:
- **~10% city-specific**: Articles with ≥50% reads from one city
  - Example: "Mumbai local election", "Bengaluru traffic update"
- **~90% NATIONAL**: Articles with distributed readership
  - Example: "India GDP growth", "Cricket World Cup"

---

## 6. Collaborative Filtering Features

**Purpose**: Enable user-user similarity recommendations ("users who read X also read Y")

**Method**: Build interaction matrix and compute user-user cosine similarity

**Optimization**: Only compute for users with ≥10 interactions to reduce noise and computational cost

---

### 1. Overview

**Key Characteristics:**
- **User-Item Matrix**: 8,560 users × 14,187 articles (sparse CSR format, 98.18% sparse)
- **User-User Similarity**: 6,589 users with ≥10 interactions (77.0% of train users)
- **Average Neighbors**: ~49 similar users per user (K=50, filtered by similarity > 0.1)
- **Engagement Weighting**: TimeSpent=1.0, Bookmarked=3.0, Shared=5.0
- **Cold Start Handling**: 1,971 users (<10 interactions) fall back to content-based only

---

### 2. Problem Statement

**Challenge:** How to recommend articles based on behavior of similar users?

**Data Characteristics:**
- **Interaction sparsity**: 98.18% of user-article pairs have no interaction
- **Power law distribution**:
  - 6,589 users (77.0%) have ≥10 interactions to eligible for collaborative filtering
  - 1,971 users (23.0%) have <10 interactions to too noisy, use content-based only
- **Engagement diversity**: Different event types signal different interest levels
  - TimeSpent-Front: 1.0 (casual browsing)
  - TimeSpent-Back: 2.0 (return visit, higher interest)
  - Bookmarked: 3.0 (explicit save)
  - Shared: 5.0 (highest engagement)

**Cold Start Issue:**
- New users have no interaction history to cannot compute similarity
- Testing articles have no historical interactions to collaborative filtering is not applicable for these items due to the absence of interaction history

---

### 3. Source Data

#### Interaction Events
- **File**: `assignment/data/processed/events.csv`
- **Shape**: 3,544,161 interactions
- **Key Columns**: `deviceId` (user), `hashId` (article), `event_type`
- **Unique Users**: 8,560 (in train_split used for CF)
- **Unique Articles**: 14,187 (in train_split used for CF)

#### Event Type Distribution
| Event Type | Count | Percentage | Engagement Weight |
|-----------------|------------|------------|-------------------|
| TimeSpent-Front | 3,500,000+ | ~98.8% | 1.0 |
| TimeSpent-Back | ~30,000 | ~0.8% | 2.0 |
| News Bookmarked | ~10,000 | ~0.3% | 3.0 |
| News Shared | ~4,000 | ~0.1% | 5.0 |

---

### 4. Feature Engineering Process

#### Step 1: Build User-Item Interaction Matrix

**Input**: events.csv (deviceId, hashId, event_type)

**Process**:
1. Map user IDs to matrix row indices (0 to 8,976)
2. Map article IDs to matrix column indices (0 to 14,187)
3. Weight each interaction by event type (TimeSpent=1.0, Bookmarked=3.0, Shared=5.0)
4. Aggregate engagement per user-article pair: `sum(event_weights)`
5. Create sparse CSR matrix (only stores non-zero values)

**Output**: `interaction_matrix.pkl`
- **Shape**: 8,560 × 14,187
- **Non-zero entries**: 2,207,948
- **Sparsity**: 98.18%
- **Format**: Compressed Sparse Row (CSR) for efficient row slicing

**Example**:
```python
# User A's interactions:
# - Read article 1 (TimeSpent-Front) → engagement = 1.0
# - Bookmarked article 2 → engagement = 3.0
# - Shared article 3 → engagement = 5.0

interaction_matrix[user_A_idx, :] = [1.0, 3.0, 5.0, 0, 0, ..., 0]
```

#### Step 2: Filter Eligible Users

**Threshold**: ≥10 interactions per user

**Rationale**:
- Users with <10 interactions have insufficient signal for reliable similarity
- Reduces noise in similarity computation
- Saves computation (6,589 vs 8,560 users = 23% reduction)

**Result**:
- **Eligible users**: 6,589 (77.0%)
- **Fallback users**: 1,971 (23.0%) to use content-based recommendations only

#### Step 3: Compute User-User Similarity

**Method**: Cosine similarity on interaction vectors

**Process**:
1. Extract eligible user rows from interaction matrix to 6,589 × 14,187
2. Compute pairwise cosine similarity to 6,589 × 6,589 similarity matrix
3. For each user, find top-K=50 most similar neighbors
4. Filter neighbors with similarity > 0.1 (remove weak matches)
5. Store as dictionary: `{user_id: [(neighbor1, sim1), (neighbor2, sim2), ...]}`

**Output**: `user_similarity.pkl`
- **Entries**: 6,589 users
- **Average neighbors per user**: ~48.7
- **Similarity range**: 0.1 to 1.0
- **Format**: Dict[str, List[Tuple[str, float]]]

**Example**:
```python
user_similarity_dict = {
    'user_A': [
        ('user_B', 0.85), # Very similar (read 85% overlapping articles)
        ('user_C', 0.72), # Similar
        ('user_D', 0.45), # Somewhat similar
        ... # up to 50 neighbors
    ],
    ...
}
```

---

### 5. Coverage Analysis

#### User Segmentation by Interaction Count

| Segment | Interaction Count | Users | Percentage | Collaborative Filtering |
|-------------------|-------------------|-------|------------|-------------------------|
| **Power Users** | ≥100 | 1,234 | 13.7% | Yes (strong signal) |
| **Active Users** | 50-99 | 1,508 | 16.8% | Yes (good signal) |
| **Regular Users** | 10-49 | 3,600 | 40.1% | Yes (adequate signal) |
| **Passive Users** | <10 | 1,971 | 23.0% | No (use content-based) |

**Key Insights**:
- 77.0% users have enough data for collaborative filtering
- 23.0% users fall back to content-based (cold start scenario)
- Power users (≥100 interactions) drive collaborative recommendations

---

### 6. Collaborative Filtering Logic

**How it works**:
1. For user U, find their top-K=50 similar neighbors
2. For each article A that U hasn't read:
   - Check which neighbors have read A
   - Weight their engagement by similarity: `score = Σ(neighbor_similarity × neighbor_engagement)`
3. Rank articles by weighted collaborative score
4. Blend with popularity and geographic signals (20% + 10% weights)

**Scoring Formula**:
```python
collaborative_score = (
    neighbor_ratings × 0.70 + # Weighted by similarity
    popularity × 0.20 +
    geographic_relevance × 0.10
)
```

**Example**:
```python
User A wants recommendations
Similar users: User B (sim=0.85), User C (sim=0.72)

Article X:
- User B read and bookmarked it (engagement=3.0)
- User C read and shared it (engagement=5.0)

Collaborative score for Article X:
= 0.85 × 3.0 + 0.72 × 5.0 = 2.55 + 3.60 = 6.15
```

---

### 7. Implementation Details

#### ID Mappings
- **File**: `mappings.pkl`
- **Contains**:
  - `user_to_idx`: {deviceId to matrix row index}
  - `article_to_idx`: {hashId to matrix column index}
  - `idx_to_user`: {matrix row index to deviceId}
  - `idx_to_article`: {matrix column index to hashId}

**Purpose**: Convert between string IDs and matrix indices for efficient computation

#### Sparse Matrix Storage

**Why CSR (Compressed Sparse Row)**:
- Only stores non-zero values to saves memory (98.18% sparse)
- Efficient row slicing to fast user vector extraction
- Optimized for matrix operations to fast similarity computation

**Memory Savings**:
- Dense matrix: 8,560 × 14,187 × 8 bytes = ~973 MB
- Sparse CSR: ~20 MB (48x reduction)

---

### 8. Example Calculations

#### Example 1: Power User with Strong Collaborative Signal

**User Profile**:
- Interactions: 150 (power user)
- Similar neighbors: 50 (all with similarity > 0.3)
- Top neighbor similarity: 0.92

**Article Recommendation**:
```python
Article A: Read by 30 neighbors (avg engagement=2.5, avg similarity=0.75)
Article B: Read by 5 neighbors (avg engagement=4.0, avg similarity=0.85)

Collaborative Score A = Σ(0.75 × 2.5) for 30 neighbors ≈ 56.25
Collaborative Score B = Σ(0.85 × 4.0) for 5 neighbors ≈ 17.00

Result: Article A ranked higher (more neighbors read it)
```

#### Example 2: Passive User (Cold Start)

**User Profile**:
- Interactions: 5 (passive user, <10 threshold)
- Similar neighbors: 0 (excluded from collaborative filtering)
- Fallback: Content-based recommendations only

**Recommendation Strategy**:
- Use category TF-IDF similarity
- Boost with popularity and language match
- No collaborative signal available

---

### 9. Impact Assessment

#### When Collaborative Filtering Helps

**Scenario 1: Discovery of New Categories**
- User typically reads "politics"
- Similar users also read "technology" (high engagement)
- Collaborative filtering recommends technology articles
- Result: Serendipitous discovery, category expansion

**Scenario 2: Quality Signal from Crowd**
- Two similar category articles (both "sports")
- Article A: High collaborative score (many neighbors read + bookmarked)
- Article B: Low collaborative score (few neighbors read)
- Result: Collaborative helps identify quality within category

**Scenario 3: Trending Content Detection**
- Breaking news article (new, no popularity history)
- Many similar users reading it right now
- Collaborative score spikes immediately
- Result: Faster trend detection than content-based alone

#### When Content-Based is Better

**Scenario 1: Cold Start Users**
- New user, no interaction history
- Cannot compute similarity
- Content-based uses language/category preferences

**Scenario 2: Niche Interests**
- User reads rare categories (e.g., "cryptocurrency")
- Few similar users with this interest
- Content-based better matches niche topics

**Scenario 3: Cold Start Articles**
- Testing set: 970 new articles with zero historical interactions
- Collaborative filtering is not applicable for these items due to the absence of interaction history
- Content-based performs reliably (uses metadata)

---

### 10. Key Decisions Rationale

#### Decision 1: K=50 Neighbors (vs K=20 or K=100)

- **Chosen**: K=50
- **Rejected**: K=20 (too few, limited diversity), K=100 (too many, weak neighbors dilute signal)
- **Why**:
  - Balances diversity and quality
  - Average ~48.7 neighbors found (after 0.1 similarity filter)
  - Provides robust collaborative signal without noise

#### Decision 2: Similarity Threshold 0.1 (vs 0.0 or 0.3)

- **Chosen**: 0.1 (filter out very weak matches)
- **Rejected**: 0.0 (include all neighbors, too noisy), 0.3 (too strict, lose diversity)
- **Why**:
  - Removes random overlap (similarity < 0.1)
  - Preserves diverse recommendations (not only very similar users)
  - Empirically good balance

#### Decision 3: ≥10 Interaction Threshold (vs ≥5 or ≥20)

- **Chosen**: ≥10
- **Rejected**: ≥5 (too noisy), ≥20 (excludes too many users)
- **Why**:
  - 10 interactions provide sufficient signal for similarity
  - Includes 77.0% of users (good coverage)
  - Reduces noise from sparse interaction vectors

#### Decision 4: Engagement Weighting (vs Binary Interaction)

- **Chosen**: TimeSpent=1.0, Bookmarked=3.0, Shared=5.0
- **Rejected**: Binary (all interactions = 1.0, ignores engagement strength)
- **Why**:
  - Captures engagement intensity (share > bookmark > read)
  - Improves recommendation quality (similar users who bookmarked have stronger signal)
  - Aligns with implicit feedback theory

---

### 11. Architecture Comparison

| Feature | User-Item Matrix | User-User Similarity |
|---------------- |----------------------------------------|----------------------------------------|
| **Purpose** | Store engagement history | Find similar users |
| **Shape** | 8,560 × 14,187 | 6,589 entries (dict) |
| **Sparsity** | 98.18% | N/A (top-K neighbors only) |
| **Storage** | ~20 MB (sparse CSR) | ~5 MB (dict) |
| **Computation** | One-time (during feature engineering) | One-time (during feature engineering) |
| **Usage** | Algorithm #2 (collaborative filtering) | Algorithm #2 (collaborative filtering) |
| **Cold Start** | Requires user interaction history | Requires ≥10 interactions |

---

### 12. Memory and Performance

#### Storage Requirements
- **interaction_matrix.pkl**: ~20 MB (sparse CSR format)
- **user_similarity.pkl**: ~5 MB (dict with top-K neighbors)
- **mappings.pkl**: <1 MB (ID mappings)

#### Computation Time
- **Interaction matrix creation**: ~10 seconds
- **User-user similarity**: ~2 minutes (6,589 × 6,589 cosine similarity)
- **Top-K neighbor selection**: ~30 seconds

#### Inference Performance
- **Per-user recommendation**: ~50-100ms
- **Batch recommendation** (3,000 users): ~30 seconds
- **Bottleneck**: Neighbor rating aggregation

---

### 13. Top User Segments

| Segment | Users | Avg Interactions | Avg Neighbors | Collaborative Quality |
|-----------------------|-------|------------------|---------------|------------------------------|
| Power Users (≥100) | 1,234 | 185 | 49.8 | Excellent (strong signal) |
| Active Users (50-99) | 1,508 | 68 | 49.2 | Very Good |
| Regular Users (10-49) | 3,600 | 22 | 48.1 | Good |
| Passive Users (<10) | 1,971 | 4 | 0 | N/A (content-based fallback) |

**Key Insight**: All eligible users (≥10 interactions) have similar neighbor counts (~48-50), indicating robust similarity computation across segments.

---

## File Structure

```
assignment/
├── data/
│ ├── processed/
│ │ ├── training_content.csv # 8,170 articles × 12 columns
│ │ ├── testing_content.csv # 970 articles × 12 columns
│ │ └── events.csv # User interactions (14,622 articles)
│ └── features/
│ ├── training_tfidf.pkl # 8,170 × 31 (articles × categories)
│ ├── testing_tfidf.pkl # 970 × 31 (articles × categories)
│ ├── user_category_tfidf.pkl # 8,689 × 31 (users × categories)
│ ├── tfidf_vectorizer.pkl # Fitted vectorizer (31 features, min_df=2)
│ ├── article_features.csv # 9,140 articles × 14 features
│ ├── user_profiles.csv # 8,689 users × 7 features
│ ├── article_locations.pkl # 13,702 location mappings
│ ├── article_popularity.pkl # 14,622 popularity scores
│ └── mappings.pkl # Various ID mappings
```

---
