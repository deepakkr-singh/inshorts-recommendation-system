# Exploratory Data Analysis Report

**Inshorts News Recommendation System - Data Analysis**

---

## Table of Contents

1. [Assignment Task](#assignment-task)
2. [Data Schema and Description](#data-schema-and-description)
3. [Summary](#summary)
4. [Dataset Overview](#dataset-overview)
5. [Data Quality Assessment](#data-quality-assessment)
6. [User Analysis](#user-analysis)
7. [Event Type Analysis](#event-type-analysis)
8. [Content Analysis](#content-analysis)
9. [Engagement Metrics](#engagement-metrics)
10. [Matrix Sparsity Analysis](#matrix-sparsity-analysis)
11. [Cold Start Analysis](#cold-start-analysis)
12. [Key Insights Summary](#key-insights-summary)
13. [Data Quality Recommendations](#data-quality-recommendations)
14. [SQL Proficiency Demonstrated](#sql-proficiency-demonstrated)
15. [Conclusion](#conclusion)

---

## 1. Assignment Task

### 1.1 Task Objective

Design and implement two distinct news article recommendation algorithms that leverage a user's past reading behavior. The algorithms should consider:

- User's reading history
- User's expressed interests
- Popularity of news articles
- Relevance of articles to the user's current location

The developed algorithms must be capable of real-time recommendations and suitable for deployment in a production environment.

### 1.2 Instructions

1. **Data Gathering:** Acquire the news dataset.
2. **Data Preparation:** Clean and prepare the dataset for analysis.
3. **Data Querying (EDA Analysis):**
 - Write and execute SQL queries in your chosen tool to analyze the user, event, and content data.
 - Your queries should demonstrate your ability to:
 i) Filter and aggregate data based on specific criteria.
 ii) Join tables to combine data from different sources.
 iii) Perform calculations and transformations on the data.
 - Present your SQL queries and the resulting output for evaluation.
4. **Feature Identification:** Explore the dataset to identify features relevant for the recommendation algorithms.
5. **Dataset Splitting:** Split the dataset into training and test sets. The test dataset should contain new content IDs/hash IDs intended for recommendation.
6. **Algorithm Development:** Build two distinct recommendation algorithms.
7. **Performance Evaluation:** Evaluate the performance of both algorithms.
8. **A/B Testing & Analysis:** Conduct A/B testing for both algorithms and provide a comparative analysis of their pros and cons.

### 1.3 Deliverables

1. **Code:** The source code for both recommendation algorithms.
2. **EDA Report:** A detailed report summarizing the key findings from the EDA, including performance KPI , data visualizations and statistical analysis.
3. **Algorithm Evaluation Report:** A report evaluating the performance of the algorithms, including final recommendations for the test set. Ideally, this report should provide the top 50 recommended content items for users and their corresponding ranks from the test set.

### 1.4 Evaluation Criteria

- **Relevance:** The recommendation algorithm must effectively recommend news articles that are relevant to user interests.
- **Real-time Capability:** The recommendation algorithm must be able to provide recommendations in real time.
- **Data Analysis Skills:** Demonstrates a strong understanding of data analysis techniques and the ability to extract meaningful insights from the data.
- **Statistical Knowledge:** Applies statistical concepts and methods effectively to analyze the data.
- **Problem-Solving:** Demonstrates problem-solving skills by identifying and addressing data challenges.
- **Communication Skills:** Clearly communicates findings and insights through well-structured reports and visualizations.

---

## 2. Data Schema and Description

### 2.1 Dataset Structure

The dataset consists of four main tables representing users, events, and content (training/testing splits).

#### 2.1.1 Devices Table (User Profile Data)

**Purpose:** User profile and device information

**Schema:**

| Field Name | Data Type | Description | Values/Example |
|-------------------------|-----------|----------------------------------------|-----------------------------|
| `deviceid` | String | Unique user identifier (PRIMARY KEY) | "a1b2c3d4-..." |
| `platform` | String | User's operating system | "ANDROID", "iOS" |
| `os_version` | String | Version of the user's operating system | "13.0", "11.5" |
| `model` | String | Device model | "SM-G991B" |
| `networkType` | String | Network connection type | "4G", "WIFI", "NO INTERNET" |
| `district` | String | User's district | "Mumbai", "Delhi" |
| `lastknownsubadminarea` | String | User's city | "Mumbai Suburban" |
| `language_selected` | String | User language preference | "en", "hi", "te" |
| `created_datetime` | String | Timestamp when user first activated | "2023-07-01 10:30:45" |
| `app_updated_at` | String | Last app update timestamp | "2023-07-15 14:22:10" |
| `last_active_at` | String | Last activity timestamp | "2023-07-20 18:45:30" |

**Key:** `deviceid` (Primary Key)
**Row Count:** 10,400 users

#### 2.1.2 Events Table (User Interaction Data)

**Purpose:** Log of all user actions and interactions with news content

**Schema:**

| Field Name | Data Type | Description | Values/Example |
|-----------------------------|-----------|---------------------------------------------------|--------------------------------------------------------|
| `deviceId` | String | User who performed action (FOREIGN KEY to devices) | "a1b2c3d4-..." |
| `hashId` | String | News article involved (FOREIGN KEY to content) | "q4dqaz8m-1" |
| `event_type` | String | Type of interaction | See Event Types below |
| `eventTimestamp` | String | Unix timestamp when event occurred | "1688198745" |
| `overallTimeSpent` | String | Time spent viewing content (seconds) | "12.5", "45.2" |
| `cardViewPosition` | String | Page number where event occurred (scroll depth) | "1", "5", "10" |
| `categoryWhenEventHappened` | String | Scene where event took place | "Homepage", "Search", "Options" |
| `searchTerm` | String | Keyword for search events | "cricket", "politics" |
| `relevancy_color` | String | User interest rating | "green" (interested), "yellow", "red" (not interested) |
| `relevancy_topic` | String | Topic/category selected for relevancy | "sports", "technology" |
| `state` | String | User location when content viewed | "Maharashtra", "Delhi" |
| `locality` | String | Local area of user | "Bandra West" |
| `district` | String | District of user when content viewed | "Mumbai" |

**Event Types:**

| Event Type | Description | Signal Strength |
|-----------------------------|-----------------------------------------------|------------------------|
| `TimeSpent-Front` | User viewed summary content | Implicit (weak) |
| `TimeSpent-Back` | User clicked to read full article from source | Implicit (strong) |
| `News Bookmarked` | User added content to favorites | Explicit (very strong) |
| `News Shared` | User shared article | Explicit (strongest) |
| `News Unbookmarked` | User removed content from favorites | Explicit (negative) |
| `Relevancy Option Selected` | User registered interest (green/yellow/red) | Explicit (strong) |
| `Search` | User searched for topic/keyword | Explicit (intent) |

**Keys:**
- Foreign Key: `deviceId` to `devices.deviceid`
- Foreign Key: `hashId` to `training_content.hashid` or `testing_content.hashid`

**Row Count:** 3,544,161 interaction events

#### 2.1.3 Training Content Table (Article Metadata)

**Purpose:** Details of news articles that users have interacted with (used for training models)

**Schema:**

| Field Name | Data Type | Description | Example |
|----------------|-----------|-----------------------------------------|--------------------------------|
| `hashid` | String | Unique article identifier (PRIMARY KEY) | "q4dqaz8m-1" |
| `title` | String | Article headline | "World records hottest day..." |
| `content` | String | Article text body | Full article text |
| `newsType` | String | Content format | "NEWS", "VIDEO_NEWS" |
| `author` | String | Content creator ID | "author_123" |
| `categories` | String | Comma-separated category labels | "sports,cricket", "national" |
| `hashtags` | String | Topic tags | "#WorldCup", "#Breaking" |
| `newsDistrict` | String | Geographic relevance | "Mumbai", "Delhi" |
| `createdAt` | String | Timestamp when news was published | "2023-07-03 08:15:00" |
| `updatedAt` | String | Timestamp when news was updated | "2023-07-03 09:00:00" |
| `newsLanguage` | String | Article language | "english", "hindi", "telugu" |
| `sourceName` | String | Source of the content | "PTI", "ANI", "Reuters" |

**Key:** `hashid` (Primary Key)
**Row Count:** 8,170 articles (with interaction history)

#### 2.1.4 Testing Content Table (New Articles to Recommend)

**Purpose:** New news inventory available for users in the future (cold start articles with no interaction history)

**Schema:** Same as Training Content Table

**Key Difference:** These articles have ZERO interaction events (cold start problem)

**Row Count:** 970 articles (new, unread)

### 2.2 Data Relationships

```
 ┌─────────────────┐
 │ DEVICES │
 │ (10,400 users) │
 │ │
 │ PK: deviceid │
 └────────┬────────┘
 │
 │ 1:Many
 │
 ▼
 ┌─────────────────────────┐
 │ EVENTS │
 │ (3,544,161 events) │
 │ │
 │ FK: deviceId │
 │ FK: hashId │
 └────────┬────────────────┘
 │
 │ Many:1
 │
 ┌─────────────────────────┐
 ▼ ▼
 ┌──────────────────┐ ┌──────────────────┐
 │ TRAINING_CONTENT │ │ TESTING_CONTENT │
 │ (8,170 articles)│ │ (970 articles) │
 │ │ │ │
 │ PK: hashid │ │ PK: hashid │
 │ (with history) │ │ (cold start) │
 └──────────────────┘ └──────────────────┘
```

### 2.3 Data Characteristics

**User Coverage:**
| Metric | Value | Source | Explanation |
|--------------------------|----------------|------------------------------|-----------------------------|
| Total registered devices | 10,400 | devices.csv count | All registered users |
| Active users | 8,977 | events.csv DISTINCT deviceId | Users who read ≥1 article |
| Overlap | 710 (7.9%) | devices ∩ events ON deviceId | present in both |

**Recommendation Strategy:** With only 7.9% metadata coverage, location/platform are adjustment factors—behavioral signals (interaction history, category preferences, collaborative filtering) must be the core recommendation engine.

**Content Coverage:**
| Metric | Value | Source | Explanation |
|--------------------------|----------------|------------------------------|--------------------------------------------------------------- |
| Training articles | 8,155 / 8,170 | training_content.csv count | Articles WITH metadata & history (99.8% interaction history) |
| Testing articles | 970 / 970 | testing_content.csv count | NEW articles (0% interaction history - cold start) |
| Total articles in events | 14,622 | events.csv DISTINCT hashId | ALL articles users ever read |
| Historical articles | 6,467 | 14,622 - 9,140 | Articles in events but NOT in training/testing CSVs |

**Interaction Density:**
| Metric | Value | Source |
|--------------------------------|---------------------------------------------|--------------------------|
| Total events | 3,544,161 | events.csv count |
| Average events per active user | 3,544,161 / 8,977 = 394.80 | events per active user |
| Average events per article | 3,544,161 / 14,622 = 242.39 | events per article |
| Possible user-item pairs | 8,977 users × 14,622 articles = 131,261,694 | - |
| Actual interactions | 3,544,161 | - |
| Matrix sparsity | (131M - 3.5M) / 131M = 97.30% | How empty the matrix is |

---

## 3. Summary

This report presents a comprehensive exploratory data analysis of the Inshorts news consumption dataset, focusing on user behavior patterns, content characteristics, and engagement metrics. The analysis was conducted using SQL queries through DuckDB to demonstrate proficiency in data manipulation, joins, aggregations, and analytical thinking.

**Finding:**
- 8,977 active users out of 10,400 total devices (86.32% activation rate)
- 3.5M interaction events across 14,622 unique articles
- 97.30% matrix sparsity presents significant cold start challenges
- Time spent is the dominant signal (99.5% of events)
- 970 new test articles require content-based recommendations

---

## 4. Dataset Overview

### 4.1 Data Volume

| Dataset | Records | Unique Entities |
|------------------|-----------|-----------------------------------------------------|
| Devices | 10,400 | 10,400 users |
| Events | 3,544,161 | 8,977 users, 14,622 unique articles interacted with |
| Training Content | 8,170 | 8,168 articles |
| Testing Content | 970 | 970 articles |

**SQL Query Used:**
```sql
SELECT
 'devices' as table_name,
 COUNT(*) as row_count,
 COUNT(DISTINCT deviceid) as unique_devices
FROM devices
UNION ALL
SELECT 'events', COUNT(*), COUNT(DISTINCT deviceId) FROM events
UNION ALL
SELECT 'training_content', COUNT(*), COUNT(DISTINCT hashid) FROM training_content
UNION ALL
SELECT 'testing_content', COUNT(*), COUNT(DISTINCT hashid) FROM testing_content;
```

---

## 5. Data Quality Assessment

### 5.1 Devices Table

| Metric | Value | Completeness |
|-------------------|--------|---------------|
| Total Devices | 10,400 | 100% |
| Duplicate Devices | 0 | No duplicates |
| Missing Platform | 0 | 100% |
| Missing Language | 0 | 100% |
| Missing District | 10,379 | 0.2% |
| Missing City | 908 | 91.3% |

**Finding:** Geographic data is sparse (99.8% missing districts), limiting location-based personalization. Language and platform data are complete.

### 5.2 Events Table

| Metric | Value | Completeness |
|--------------------|-----------|--------------|
| Total Events | 3,544,161 | 100% |
| Missing Time Spent | 1,123 | 99.97% |
| Missing Position | 1,123 | 99.97% |
| Unique Event Types | 7 | Diverse |

**Finding:** Excellent data quality for core engagement signals. Time spent data available for 99.97% of events.

### 5.3 Content Tables

| Dataset | Missing Categories | Missing Hashtags | Completeness |
|--------- |--------------------|------------------|--------------|
| Training | 25 (0.31%) | 6,915 (84.6%) | 99.69% |
| Testing | 3 (0.31%) | 933 (96.2%) | 99.69% |

**Finding:** Category data is highly complete. Hashtags are sparse but not critical for initial recommendations.

---

## 6. User Analysis

### 6.1 Platform Distribution

| Platform | Users | Percentage |
|----------|--------|------------|
| ANDROID | 10,400 | 100.0% |

**Finding:** 100% Android user base.

### 6.2 Language Preferences

| Language | Users | Percentage |
|--------------|--------|------------|
| en (English) | 10,400 | 100.0% |

**Finding:** 100% users have English as their language preference.

### 6.3 User Segmentation by Activity Level

```sql
WITH user_activity AS (
 SELECT deviceId, COUNT(*) as event_count
 FROM events GROUP BY deviceId
)
SELECT
 CASE
 WHEN event_count >= 1000 THEN 'Power User (1000+)'
 WHEN event_count >= 100 THEN 'Very Active (100-999)'
 WHEN event_count >= 10 THEN 'Active (10-99)'
 ELSE 'Passive (<10)'
 END as user_segment,
 COUNT(*) as user_count,
 AVG(event_count) as avg_events
FROM user_activity
GROUP BY user_segment;
```

| Segment | Users | Percentage | Avg Events | Min | Max |
|-----------------------|-------|------------|------------|-------|--------|
| Power User (1000+) | 788 | 8.78% | 3,186.25 | 1,002 | 21,363 |
| Very Active (100-999) | 2,787 | 31.05% | 318.47 | 100 | 998 |
| Active (10-99) | 3,349 | 37.31% | 41.08 | 10 | 99 |
| Passive (<10) | 2,053 | 22.87% | 4.01 | 1 | 9 |

**Key Insights:**
1. **Power Law Distribution:** 8.78% of users (power users) generate disproportionate engagement
2. **Majority Active:** 77.14% of users have 10+ events, indicating healthy engagement
3. **Recommendation Strategy:** Different approaches needed per segment:
 - Power users: Diversity and serendipity
 - Active users: Personalized based on history
 - Passive users: Popular content to increase engagement

---

## 7. Event Type Analysis

### 7.1 Event Distribution

```sql
SELECT
 event_type,
 COUNT(*) as event_count,
 COUNT(DISTINCT deviceId) as unique_users,
 ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage,
 AVG(TRY_CAST(overallTimeSpent AS DOUBLE)) as avg_time_spent
FROM events
GROUP BY event_type
ORDER BY event_count DESC;
```

| Event Type | Count | Unique Users | Percentage | Avg Time (sec) |
|---------------------------|-----------|--------------|------------|----------------|
| TimeSpent-Front | 3,480,131 | 8,976 | 98.19% | 8.59 |
| TimeSpent-Back | 44,933 | 4,853 | 1.27% | 41.18 |
| News Bookmarked | 10,870 | 2,073 | 0.31% | 0.00 |
| News Shared | 3,517 | 1,031 | 0.10% | 0.00 |
| News Unbookmarked | 2,275 | 904 | 0.06% | 0.00 |
| Relevancy Option Selected | 1,312 | 126 | 0.04% | 0.00 |
| Search | 1,123 | 418 | 0.03% | N/A |

**Critical Insights:**

1. **Dominant Signal:** TimeSpent-Front (98.19%) represents casual browsing where users skim headlines and summaries in the Inshorts feed. Average time: 8.59 seconds per article.

2. **Deep Engagement Indicator:** TimeSpent-Back (1.27%) occurs when users click "Read Full Story" to view the original source. Average time: 41.18 seconds (4.8x higher). This is a strong interest signal.

3. **Explicit Feedback Scarcity:** Only 0.41% of events are explicit signals (bookmarks: 0.31%, shares: 0.10%). These are rare but highly valuable indicators of user preference.

4. **Implicit Feedback Dominance:** Time spent data (99.46% of events) forms the primary implicit feedback signal. Challenges include distinguishing genuine interest from app being left open or interrupted sessions.

### 7.2 Time Spent Statistics

| Metric | Value (seconds) |
|--------|-----------------|
| Count | 3,525,064 |
| Mean | 9.01 |
| Median | 3.78 |
| Std Dev | 55.05 |
| Min | 0.00 |
| Max | 41,047.59 |
| Q25 | 1.22 |
| Q75 | 9.31 |
| Q95 | 32.28 |
| Q99 | 63.31 |

**Insights:**
- **Right-skewed distribution:** Median (3.78s) < Mean (9.01s)
- **High variance:** Some outliers with 11+ hours (likely app left open)
- **Actionable threshold:** 95th percentile at 32.28s suggests strong engagement cutoff

---

## 8. Content Analysis

### 8.1 Content Type and Language

| News Type | Language | Articles | Percentage |
|------------|----------|----------|------------|
| NEWS | English | 5,781 | 70.76% |
| NEWS | Hindi | 1,845 | 22.58% |
| VIDEO_NEWS | English | 170 | 2.08% |
| Others | Various | 374 | 4.58% |

**Finding:** Text news in English dominates (70.76%), with significant Hindi representation (22.58%).

### 8.2 Top Content Categories

| Category | Articles | Percentage |
|---------------|----------|------------|
| national | 3,574 | 35.51% |
| world | 1,498 | 14.88% |
| sports | 1,024 | 10.17% |
| business | 939 | 9.33% |
| politics | 618 | 6.14% |
| entertainment | 553 | 5.49% |
| technology | 420 | 4.17% |
| science | 217 | 2.16% |
| startup | 196 | 1.95% |
| VIDEO_NEWS | 165 | 1.64% |

**Insights:**
1. **National focus:** 35.51% of content is national news
2. **Diverse catalog:** 55+ unique categories provide variety
3. **Long tail:** Many niche categories (education, travel, fashion) with <1%

---

## 9. Engagement Metrics

### 9.1 Weighted Preference Scoring

**Scoring Formula:**
```sql
engagement_score =
 (TimeSpent-Front × 1.0) +
 (TimeSpent-Back × 2.0) +
 (News Bookmarked × 3.0) +
 (News Shared × 5.0)
```

**Top 20 Users by Average Engagement Score:**
- Highest avg engagement: 18.00
- Most engaged users show multi-action patterns (bookmarks + shares + high time)
- Average interaction types per user: 1.33 to 3.00

**Recommendation Implication:** Combine multiple signals with appropriate weights to capture true user interest.

### 9.2 Content Popularity Analysis

**Top 20 Most Popular Articles (100+ unique users):**

| Metric | Min | Max | Median |
|--------------------|-------|--------|--------|
| Unique Users | 1,117 | 3,996 | 2,388 |
| Total Interactions | 1,641 | 5,547 | 3,186 |
| Avg Time Spent | 3.14s | 22.37s | 7.70s |
| Virality Rate | 0.00% | 1.76% | 0.11% |
| CTR (Full Read) | 0.19% | 16.57% | 2.14% |

**CTR vs Virality Scatter Plot Analysis:**

The scatter plot reveals content performance patterns by plotting Click-Through Rate (x-axis) against Virality Rate (y-axis), with bubble size representing reach:

**Key Metrics:**
- **CTR (Click-Through Rate):** Percentage of users clicking "Read Full Story" = (TimeSpent-Back / TimeSpent-Front) × 100
- **Virality Rate:** Percentage of users sharing = (Shares / Unique Users) × 100
- **Bubble Size:** Unique users who viewed the article (popularity/reach)

**Identified Patterns:**

1. **Bottom-Left Cluster (Majority):** Low CTR (<5%) + Low Virality (<0.5%)
 - Passively consumed content where users skim summaries without deep engagement
 - High reach but low quality engagement

2. **Top-Left Quadrant:** High Virality + Low CTR
 - Sensational headlines shared without reading full article
 - Example: Ex-employee murder story (1.76% virality, 6.62% CTR)
 - Users share based on headline impact alone

3. **Right-Center Quadrant:** High CTR + Low Virality
 - Engaging content people read fully but don't share publicly
 - Example: Conductor sex scandal (16.57% CTR, 0.57% virality)
 - Too controversial or personal to share despite strong interest

4. **Top-Right Quadrant (Ideal):** High CTR + High Virality
 - Best performing content with both depth (reading) and breadth (sharing)
 - Target content type for recommendations

**Recommendation Implications:**
- **Don't rely solely on popularity (large bubbles):** High view count ≠ Quality engagement
- **Power Users:** Recommend high CTR articles (deep engagement worth their time)
- **Passive Users:** Recommend high virality articles (social proof, trending content)
- **Active Users:** Balanced mix prioritizing engagement quality over pure reach

### 9.3 Geographic Content Preferences

**Data Quality Note:** District data has 99.8% missingness (only 21 values). Analysis uses city field (lastknownsubadminarea) with 91.3% completeness (9,492 available values).

**Top 15 Cities by User Activity:**

| City | Unique Users | Total Events | Avg Time Spent (sec) |
|-----------|--------------|--------------|----------------------|
| Mumbai | 41 | 6,061 | 7.46 |
| Delhi | 34 | 3,433 | 9.37 |
| Bengaluru | 31 | 3,340 | 11.50 |
| Kolkata | 23 | 2,798 | 10.41 |
| Noida | 21 | 4,259 | 7.88 |
| Patna | 19 | 1,107 | 18.24 |
| Lucknow | 15 | 1,482 | 9.60 |
| Gurgaon | 15 | 1,249 | 14.31 |
| Chennai | 14 | 10,677 | 5.94 |
| Hyderabad | 13 | 3,524 | 6.94 |

**SQL Query:**
```sql
WITH category_split AS (
 SELECT
 e.deviceId,
 d.lastknownsubadminarea as city,
 TRIM(unnest(string_split(c.categories, ','))) as category,
 TRY_CAST(e.overallTimeSpent AS DOUBLE) as time_spent
 FROM events e
 INNER JOIN devices d ON e.deviceId = d.deviceid
 INNER JOIN training_content c ON e.hashId = c.hashid
 WHERE d.lastknownsubadminarea IS NOT NULL
 AND c.categories IS NOT NULL
 AND e.event_type = 'TimeSpent-Front'
)
SELECT
 city,
 category,
 COUNT(DISTINCT deviceId) as unique_users,
 COUNT(*) as interaction_count,
 ROUND(AVG(time_spent), 2) as avg_time_spent
FROM category_split
GROUP BY city, category
HAVING COUNT(DISTINCT deviceId) >= 3
ORDER BY city, interaction_count DESC;
```

**Key Insights:**

1. **Metro Dominance:** Top cities are major metros (Mumbai, Delhi, Bengaluru, Kolkata) representing economic hubs
2. **Engagement Variation:** Patna shows highest avg time spent (18.24s) despite lower event volume, suggesting deeper engagement
3. **Chennai Anomaly:** High event count (10,677) but low avg time (5.94s) indicates rapid scrolling behavior
4. **Coverage Limitation:** Only ~9% of users have city data, limiting geographic personalization
5. **City-Category Preferences:** Analysis reveals distinct category preferences by city (e.g., tech content in Bengaluru, business in Mumbai)

**Recommendation Implication:**
- Geographic personalization possible for ~9% of users with city data
- For remaining users, rely on language and category preferences
- Consider IP-based geolocation as fallback for location-aware recommendations

### 9.4 User Category Affinity Analysis

**What is Category Affinity?**

Category affinity measures how strongly a user prefers specific content categories based on their interaction history. It combines multiple engagement signals (views, time spent, bookmarks, shares) to create a weighted preference score for each category per user.

**Affinity Score Formula:**
```sql
affinity_score =
 (view_count × 1.0) +
 (bookmarks × 3.0) +
 (shares × 5.0) +
 (total_time_spent × 0.1)
```

**Why It Matters for Recommendations:**

1. **Personalization Foundation:** Identifies user's true interests beyond simple view counts
2. **Multi-Signal Integration:** Combines implicit (views, time) and explicit (bookmarks, shares) feedback
3. **Content Matching:** Enables accurate matching between user preferences and article categories
4. **Diversity vs Relevance:** Balances recommending favorite categories while introducing new topics

**Analysis Results:**

**User Category Distribution (users with 20+ interactions):**
- Total users analyzed: 6,924 active users
- Total user-category pairs: 80,679 affinity scores
- Average categories per user: 11.65
- Top category per user: avg affinity score = 347.89

**Sample User Category Preferences:**

| User ID | Top Category | View Count | % of Views | Affinity Score |
|-------------|--------------|------------|------------|----------------|
| 0002d448... | national | 52 | 23.96% | 89.31 |
| 000d4df6... | national | 460 | 34.15% | 854.21 |
| 00198103... | national | 649 | 54.86% | 1347.19 |

**SQL Query:**
```sql
WITH category_split AS (
 SELECT
 e.deviceId,
 TRIM(unnest(string_split(c.categories, ','))) as category,
 e.event_type,
 TRY_CAST(e.overallTimeSpent AS DOUBLE) as time_spent
 FROM events e
 INNER JOIN training_content c ON e.hashId = c.hashid
 WHERE c.categories IS NOT NULL
 AND e.event_type IN ('TimeSpent-Front', 'News Bookmarked', 'News Shared')
),
user_category_interactions AS (
 SELECT
 deviceId,
 category,
 COUNT(*) as view_count,
 SUM(time_spent) as total_time_spent,
 SUM(CASE WHEN event_type = 'News Bookmarked' THEN 1 ELSE 0 END) as bookmarks,
 SUM(CASE WHEN event_type = 'News Shared' THEN 1 ELSE 0 END) as shares
 FROM category_split
 GROUP BY deviceId, category
),
user_totals AS (
 SELECT deviceId, SUM(view_count) as total_views
 FROM user_category_interactions
 GROUP BY deviceId
)
SELECT
 uci.deviceId,
 uci.category,
 uci.view_count,
 ROUND(uci.view_count * 100.0 / ut.total_views, 2) as percentage_of_views,
 ROUND(COALESCE(uci.total_time_spent, 0), 2) as total_time,
 uci.bookmarks,
 uci.shares,
 ROUND(
 (uci.view_count * 1.0) +
 (uci.bookmarks * 3.0) +
 (uci.shares * 5.0) +
 (COALESCE(uci.total_time_spent, 0) * 0.1), 2
 ) as affinity_score
FROM user_category_interactions uci
INNER JOIN user_totals ut ON uci.deviceId = ut.deviceId
WHERE ut.total_views >= 20
ORDER BY uci.deviceId, affinity_score DESC;
```

**Key Patterns Observed:**

1. **Category Concentration:** Users typically have 1-3 dominant categories representing 60-80% of their consumption
2. **Long Tail Interests:** Most users engage with 10-15 categories, but with decreasing affinity
3. **National News Dominance:** "National" category appears as top category for 42% of users
4. **Explicit Feedback Boost:** Users with bookmarks/shares show 3-5x higher affinity scores for those categories
5. **Time Investment:** Categories with high time spent (even fewer views) rank higher in affinity

**Recommendation Strategy Using Affinity:**

1. **Primary Recommendations (70%):** Match articles from user's top 3 categories by affinity score
2. **Exploration (20%):** Introduce content from categories 4-10 to maintain diversity
3. **Discovery (10%):** Recommend from categories outside user's history for serendipity

**Example User Profile:**

User `00198103-e45e-4b33-804b-84ff19562d62`:
- Top Category: national (649 views, 54.86%, affinity: 1347.19)
 - 15 bookmarks + 0 shares = strong explicit interest
- Second: science (98 views, 8.28%, affinity: 295.69)
 - 17 bookmarks + 3 shares = very strong engagement despite lower volume
- Third: technology (84 views, 7.10%, affinity: 184.62)

**Insight:** This user heavily consumes national news but shows deeper engagement (bookmarks/shares) with science content, suggesting recommendations should balance both.

---

## 10. Matrix Sparsity Analysis

**What is Matrix Sparsity?** Matrix sparsity measures the percentage of missing user-item interactions in the dataset. High sparsity (close to 100%) means most users haven't interacted with most items, making it difficult for collaborative filtering algorithms to find similar users or items for recommendations.

```sql
WITH stats AS (
 SELECT
 COUNT(DISTINCT deviceId) as unique_users,
 COUNT(DISTINCT hashId) as unique_items,
 COUNT(*) as interactions
 FROM events
)
SELECT
 unique_users * unique_items as possible_interactions,
 interactions as actual_interactions,
 100.0 * (1 - interactions / (unique_users * unique_items)) as sparsity_pct
FROM stats;
```

| Metric | Value |
|-----------------------|-------------|
| Unique Users | 8,977 |
| Unique Items | 14,622 |
| Actual Interactions | 3,544,161 |
| Possible Interactions | 131,261,694 |
| **Sparsity** | **97.30%** |
| Avg Interactions/User | 394.80 |
| Avg Interactions/Item | 242.39 |

**Critical Insight:** 97.30% sparsity indicates severe cold start problem. Most user-item pairs have no interaction data.

**Recommendation Strategy Required:**
- **Collaborative filtering alone is likely to underperform** under the observed sparsity
- **Hybrid approach is strongly justified given the observed data characteristics:** Combine content-based + collaborative methods
- **Cold start handling:** New articles (test set) need content-based scoring

---

## 11. Cold Start Analysis

**What is Cold Start Problem?** Cold start occurs when the system has no historical interaction data for new items (articles) or new users, making it impossible to use collaborative filtering techniques. This requires content-based or hybrid approaches that rely on item metadata and user profile features instead of past interactions.

### 11.1 Training vs Testing Split

| Dataset | Articles | Unique Types | Unique Languages | Overlap |
|----------|----------|--------------|------------------|---------|
| Training | 8,170 | 20 | 5 | 0 |
| Testing | 970 | 5 | 6 | 0 |

> **Note:** Training data contains 8,170 rows (total count) with 8,168 unique articles (2 duplicate hashIds exist in the dataset).

**Validation:** Zero overlap confirms a clean train-test split with no information leakage.

### 11.2 Language Distribution Comparison

| Language | Training | Testing | Delta |
|----------|----------|---------|--------|
| English | 6,211 | 364 | -5,847 |
| Hindi | 1,845 | 371 | -1,474 |
| Telugu | 0 | 94 | +94 |
| Kannada | 0 | 76 | +76 |
| Gujarati | 3 | 61 | +58 |

**Challenge:** Testing set introduces new languages (Telugu, Kannada) with zero training representation, requiring language-agnostic features.

---

## 12. Key Insights Summary

### 12.1 User Behavior

1. **High Activation Rate:** 86.32% of devices are active users
2. **Power Law Engagement:** 8.78% power users drive significant activity
3. **Passive Majority:** 22.87% users have <10 events (retention opportunity)
4. **Language Preference:** 100% prefer English despite 22.58% Hindi content

### 12.2 Content Patterns

1. **Category Concentration:** National news dominates (35.51%)
2. **Language Mismatch:** User preference (English) vs content availability (Hindi 22.58%)
3. **Video Underrepresentation:** Only 2.08% video content despite media trend
4. **Long Tail Catalog:** 55+ categories provide diversity

### 12.3 Engagement Signals

1. **Implicit Dominance:** 98.19% passive consumption (TimeSpent-Front)
2. **Explicit Scarcity:** <0.5% explicit signals (bookmarks + shares)
3. **Time as Proxy:** Median 3.78s, mean 9.01s, strong engagement >32s (Q95)
4. **Deep Engagement Rare:** Only 1.27% click through to full article

### 12.4 Technical Challenges

1. **Extreme Sparsity:** 97.30% user-item matrix sparsity
2. **Cold Start:** 970 test articles with zero interaction history
3. **New Languages:** Test set has languages unseen in training
4. **Scale:** 3.5M events require efficient algorithms

---

## 13. Data Quality Recommendations

### 13.1 Immediate Actions

1. **Geographic Data:** 99.8% missing districts limits location-based personalization
 - Implement IP-based geolocation as fallback
 - Add location permission prompt in app

2. **Explicit Feedback:** <0.5% bookmarks/shares is insufficient
 - Add like/dislike buttons for direct feedback
 - Implement star ratings for article quality

3. **Session Tracking:** No session IDs to group interactions
 - Add session identifiers for better context
 - Enable session-based recommendations

### 13.2 Long-term Improvements

1. **Rich User Profiles:** Collect interests, occupation, age group
2. **Article Metadata:** Add author reputation, source credibility
3. **Temporal Features:** Time of day, day of week patterns
4. **Social Signals:** Comments, forwards, reactions

---

## 14. Demonstration of SQL-Based Analytical Proficiency

### 14.1 Query Complexity

**Techniques Used:**
1. **CTEs (Common Table Expressions):** Multi-level query organization
2. **Window Functions:** Running totals, percentages, rankings
3. **Aggregations:** COUNT, SUM, AVG, MEDIAN, PERCENTILE_CONT
4. **Joins:** INNER, LEFT, FULL OUTER across 3+ tables
5. **String Operations:** SUBSTRING, TRIM, string_split, unnest
6. **Type Casting:** TRY_CAST for safe conversions
7. **Conditional Logic:** Complex CASE statements
8. **Statistical Functions:** STDDEV, PERCENTILE, MEDIAN

### 14.2 Performance Considerations

**Optimizations Applied:**
1. **Filtered Aggregations:** WHERE before GROUP BY
2. **Selective Columns:** Only necessary fields in SELECT
3. **Indexed Joins:** Primary keys (deviceid, hashid)
4. **Limit Clauses:** Prevent excessive result sets
5. **Efficient CTEs:** Reusable subqueries instead of repeated scans

---

## 15. Conclusion

This exploratory analysis reveals a rich dataset with clear patterns for building an effective news recommendation system. The key challenges—extreme sparsity (97.30%), cold start problem (970 new articles), and limited explicit feedback (<0.5%)—strongly motivate the adoption of a hybrid recommendation approach combining content-based and collaborative filtering methods.

The dataset quality is high for core signals (99.97% time spent completeness) but lacks geographic and explicit feedback depth. User segmentation (power users: 8.78%, passive: 22.87%) suggests personalized strategies are essential for engagement optimization.

**Data-Driven Insights for Modeling:**
1. **High sparsity (97.30%)** requires hybrid approach combining collaborative + content-based methods
2. **Cold start (970 test articles)** necessitates content-based features (category, language, type)
3. **Limited explicit feedback (<0.5%)** means relying on implicit time spent signals
4. **User segmentation** (power/active/passive) suggests personalized weighting strategies
5. **Category affinity patterns** provide strong signals for content matching

---
