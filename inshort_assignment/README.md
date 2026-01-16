# Inshorts Recommendation System

## Project Overview
Real-time news recommendation system using hybrid approach (Collaborative Filtering + Content-Based Filtering) with **2ms latency** and **526 users/sec throughput**.

## Project Structure

### `/docs/` - Documentation (5 Reports)
1. **[01_eda_report.md](inshort_assignment/docs/01_eda_report.md)** - Exploratory Data Analysis with SQL queries
2. **[02_model_selection_report.md](inshort_assignment/docs/02_model_selection_report.md)** - Algorithm selection rationale
3. **[03_feature_engineering_pipeline.md](inshort_assignment/docs/03_feature_engineering_pipeline.md)** - Feature generation process
4. **[04_performance_evaluation_and_ab_testing.md](inshort_assignment/docs/04_performance_evaluation_and_ab_testing.md)** - Metrics & A/B testing
5. **[05_production_readiness_and_realtime_capability.md](inshort_assignment/docs/05_production_readiness_and_realtime_capability.md)** - Production deployment analysis

### `/notebooks/` - Jupyter Notebooks (3 Files)
1. **[01_eda_sql.ipynb](inshort_assignment/notebooks/01_eda_sql.ipynb)** - SQL-based data exploration (30+ queries)
2. **[02_feature_engineering.ipynb](inshort_assignment/notebooks/02_feature_engineering.ipynb)** - User profiles & article features
3. **[03_evaluation.ipynb](inshort_assignment/notebooks/03_evaluation.ipynb)** - Performance evaluation (7 metrics)

### `/data/` - Generated Data Files
- **`processed/`** - Cleaned datasets (train/val/test splits)
- **`features/`** - User profiles & article features
- **`recommendations/`** - Final recommendation outputs
  - **[`final_recommendations.csv`](inshort_assignment/data/recommendations/final_recommendations.csv)**  **Top 50 recommendations** for each user with ranks & scores (8,689 users × 50 items = 434,450 rows)
  - `collaborative_recommendations.csv` - CF-based recommendations
  - `content_based_recommendations.csv` - Content-based recommendations
  - `hybrid_recommendations.csv` - Hybrid approach recommendations
- **`evaluation/`** - Performance metrics & evaluation results
  - **[`test_recommendations.csv`](inshort_assignment/data/evaluation/test_recommendations.csv)** - Test set recommendations with feature scores

### `/src/` - Source Code
- **[data_loader.py](inshort_assignment/src/data_loader.py)** - GCS data loading utilities

### `/test_data/` - Raw Input Data
- Original datasets from GCS (devices, events, content)

## Quick Start

### 1. Run EDA
```bash
jupyter notebook inshort_assignment/notebooks/01_eda_sql.ipynb
```

### 2. Generate Features
```bash
jupyter notebook inshort_assignment/notebooks/02_feature_engineering.ipynb
```

### 3. Evaluate Models
```bash
jupyter notebook inshort_assignment/notebooks/03_evaluation.ipynb
```

## Key Metrics

| Metric | Value |
|--------|-------|
| **Latency** | ~2ms per user |
| **Throughput** | 526 users/second |
| **Memory** | 26MB core data |
| **NDCG@50** | 0.112 (collaborative) |
| **Precision@50** | 11.35% (engaged users) |

## Navigation Guide

**Want to understand the data?**  
→ Start with [01_eda_report.md](inshort_assignment/docs/01_eda_report.md) & [01_eda_sql.ipynb](inshort_assignment/notebooks/01_eda_sql.ipynb)

**Want to see algorithm selection?**  
→ Read [02_model_selection_report.md](inshort_assignment/docs/02_model_selection_report.md)

**Want to see feature engineering?**  
→ Check [03_feature_engineering_pipeline.md](inshort_assignment/docs/03_feature_engineering_pipeline.md) & [02_feature_engineering.ipynb](inshort_assignment/notebooks/02_feature_engineering.ipynb)

**Want to see performance results?**  
→ Open [04_performance_evaluation_and_ab_testing.md](inshort_assignment/docs/04_performance_evaluation_and_ab_testing.md) & [03_evaluation.ipynb](inshort_assignment/notebooks/03_evaluation.ipynb)

**Want to see production readiness?**  
→ Read [05_production_readiness_and_realtime_capability.md](inshort_assignment/docs/05_production_readiness_and_realtime_capability.md)

## Key Deliverables

✅ **2 Algorithms**: Collaborative Filtering + Content-Based  
✅ **7 Metrics**: NDCG, Precision, Recall, MAP, Coverage, Diversity, Novelty  
✅ **Real-time**: <100ms requirement (achieved 2ms)  
✅ **Top 50 Recommendations**: Generated for all 8,689 users → **[final_recommendations.csv](inshort_assignment/data/recommendations/final_recommendations.csv)** (434,450 rows with deviceId, hashId, rank, score)  
✅ **Production-ready**: Comprehensive deployment analysis  

## Performance Highlights

- **50x faster** than required latency (<100ms → 2ms)
- **526 users/sec** parallel processing capability
- **99%+ memory efficiency** with sparse matrices
- **Near-complete coverage** with fallback strategies

---
