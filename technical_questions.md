# Technical Questions - Interview Preparation

## Data Loader Questions

### 1. Why did you use `on_bad_lines="skip"` when reading CSVs?

Real-world data often contains malformed rows due to inconsistent formatting or special characters. Using `on_bad_lines="skip"` prevents the entire load from failing due to a few corrupted rows. For production systems, I would enhance this by logging skipped rows to a separate file for manual review and data quality monitoring.

### 2. Why validate columns before type conversion?

This follows the fail fast principle. By validating the schema upfront, we catch structural issues immediately with clear error messages rather than getting cryptic type conversion errors later in the pipeline. This makes debugging much faster and provides better error messages for data quality issues.

### 3. Why use `pd.to_datetime(..., errors='coerce')` instead of raising errors?

News data from multiple sources can have inconsistent timestamp formats (ISO 8601, Unix timestamps, various date formats). The coerce strategy converts invalid dates to NaT (Not a Time) which allows processing to continue while flagging data quality issues through our null detection warnings. This is pragmatic for exploratory analysis while maintaining data quality awareness.

### 4. How would you handle the 3.5M event rows if memory was limited?

Several approaches depending on constraints:
- Use pandas `chunksize` parameter to process data in batches
- Switch to Polars for more efficient memory usage with lazy evaluation
- Use Dask for distributed processing on larger-than-memory datasets
- Leverage DuckDB for out-of-core SQL operations
- For production, consider streaming architectures with Apache Spark or similar

### 5. Why separate raw and processed paths?

This follows data engineering best practices:
- Raw data remains immutable and serves as the source of truth
- Processed data can be regenerated if pipeline logic changes
- Enables reproducibility and debugging
- Supports data lineage tracking
- Prevents accidental corruption of source data

### 6. The event data is loaded from multiple part files. Why?

This is standard output from distributed processing systems like Apache Spark or Hadoop. Each partition writes to a separate file. Using glob patterns (`part-*.csv`) handles any number of partitions automatically without hardcoding filenames, making the loader robust to changes in partition count.

## Code Quality Questions

### 7. Why use Path instead of string concatenation?

Path objects from pathlib provide several advantages:
- Cross-platform compatibility (handles Windows vs Unix path separators)
- Cleaner, more readable API with `/` operator
- Built-in methods for path manipulation
- Eliminates common bugs from manual string concatenation
- Type safety and better IDE support

### 8. Why static methods for helpers?

The validation and logging helpers are pure functions with no dependency on instance state. Making them static methods provides several benefits:
- Easier to unit test in isolation
- Clearer intent (no side effects on instance)
- Can be called without instantiating the class
- Better code organization and reusability

### 9. How would you add logging instead of print statements?

Replace print statements with Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

logger.warning(f"{name}: nulls detected -> {nulls.to_dict()}")
logger.info("Data loaded and validated successfully")
```

Configure logging levels (DEBUG, INFO, WARN, ERROR) for different environments. In production, send logs to centralized logging systems like ELK stack or CloudWatch.

### 10. How would you test this class?

Testing strategy:
- Use pytest with fixtures for sample CSV data
- Mock file I/O operations to avoid dependency on actual files
- Test validation logic separately with edge cases
- Test error handling (missing files, malformed data, wrong schema)
- Integration tests with small real data samples
- Use pytest-cov to ensure adequate code coverage

Example test structure:
```python
def test_validate_columns_missing():
    df = pd.DataFrame({'col1': [1, 2]})
    with pytest.raises(ValueError):
        DataLoader._validate_columns(df, ['col1', 'col2'], 'test')
```

## Design Questions

### 11. Why not use DuckDB here for loading?

Separation of concerns:
- DuckDB excels at analytical queries (SQL-based EDA)
- Pandas is simpler for ETL and data transformations
- Pandas integrates seamlessly with scikit-learn for modeling
- DuckDB introduces additional complexity for simple CSV operations
- Keep loading logic simple and use DuckDB where SQL provides clear value

### 12. What would you change for production?

Production enhancements:
- Add comprehensive logging with log levels and structured logging
- Implement retry logic with exponential backoff for network issues
- Add data quality metrics (completeness, accuracy, timeliness)
- Schema validation using Pydantic or Great Expectations
- Monitoring and alerting for data pipeline failures
- Version control for data schemas
- Add data profiling and statistics collection
- Implement data lineage tracking
- Add unit tests and integration tests with CI/CD
- Use configuration files instead of hardcoded paths
- Add data validation checkpoints between pipeline stages

## SQL and DuckDB Questions

### 13. When would you use DuckDB vs Pandas vs Spark?

**DuckDB:**
- SQL-based analytics on medium datasets (GBs)
- Fast aggregations and joins without distributed setup
- When SQL expressiveness is preferred
- Embedded analytics in applications

**Pandas:**
- Interactive data exploration
- Integration with ML libraries
- Complex transformations requiring Python flexibility
- Datasets that fit in memory

**Spark:**
- Large-scale distributed processing (TBs+)
- Production data pipelines
- When horizontal scaling is required
- Integration with Hadoop ecosystem

### 14. How does this loader support the recommendation system requirements?

Design decisions aligned with requirements:
- Loads all four datasets (devices, events, training, testing content)
- Preserves data types needed for feature engineering
- Handles large event dataset efficiently
- Separates training and testing content for proper evaluation
- Maintains data quality through validation
- Creates reproducible processed datasets for modeling

## Feature Engineering Questions

### 15. What features would you derive from this raw data?

User features:
- Activity level segmentation
- Language preferences
- Geographic location
- Device type patterns

Content features:
- Category distributions
- Popularity metrics
- Recency scores
- Language and type

Interaction features:
- Time spent signals
- Explicit feedback (bookmarks, shares)
- Implicit feedback (views, scroll depth)
- Temporal patterns

### 16. How would you handle the cold start problem visible in the testing set?

The testing content has no interaction history. Approaches:
- Content-based filtering using article metadata (categories, language, type)
- Popularity-based recommendations for new users
- Hybrid approach combining collaborative and content-based
- Use training set to learn content preferences
- Geographic and language matching
- Leverage user profiles from device data