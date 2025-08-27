# Data Loading and Management

Pradel-JAX provides flexible data loading capabilities that can handle multiple data formats commonly used in capture-recapture studies. The framework automatically detects data formats and provides comprehensive validation to ensure data quality.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Data Formats](#supported-data-formats)
3. [Data Loading API](#data-loading-api)
4. [Data Validation](#data-validation)
5. [Working with Large Datasets](#working-with-large-datasets)
6. [Time-Varying Covariates](#time-varying-covariates)
7. [Data Sampling and Subsets](#data-sampling-and-subsets)
8. [Troubleshooting](#troubleshooting)

## Quick Start

```python
import pradel_jax as pj

# Load data from any supported format
data_context = pj.load_data("path/to/your/data.csv")

# Check what was loaded
print(f"Loaded {data_context.n_individuals} individuals")
print(f"Capture occasions: {data_context.n_occasions}")
print(f"Available covariates: {list(data_context.covariates.keys())}")

# Validate data quality
validation_result = data_context.validate()
if validation_result.is_valid:
    print("✅ Data validation passed!")
else:
    print("⚠️ Data validation issues:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

## Supported Data Formats

### 1. RMark Format (Recommended)

The most common format used in capture-recapture studies, typically exported from MARK software.

**Structure:**
- `ch` column: Capture history string (e.g., "11010")
- Individual covariates as additional columns
- Each row represents one individual

**Example:**
```csv
ch,sex,age,weight
11010,M,adult,2.5
10101,F,juvenile,1.8
01011,M,adult,2.2
```

**Loading:**
```python
data_context = pj.load_data("rmark_data.csv")
# Automatically detects RMark format based on 'ch' column
```

### 2. Y-Column Format

Alternative format where each capture occasion has its own column.

**Structure:**
- `Y2016`, `Y2017`, etc.: Binary capture indicators (0/1)
- Individual covariates as additional columns
- Each row represents one individual

**Example:**
```csv
Y2016,Y2017,Y2018,Y2019,Y2020,sex,age
1,1,0,1,0,M,adult
1,0,1,0,1,F,juvenile
0,1,0,1,1,M,adult
```

**Loading:**
```python
data_context = pj.load_data("y_column_data.csv")
# Automatically detects Y-column format based on column names
```

### 3. Generic Format

Flexible format for custom data structures.

**Structure:**
- Capture columns can have any names
- Covariates can be mixed with capture data
- Requires explicit column specification

**Example:**
```csv
ID,capture_2016,capture_2017,capture_2018,sex,region,tier
1,1,0,1,M,north,1
2,0,1,1,F,south,2
3,1,1,0,M,north,1
```

**Loading:**
```python
# Specify capture columns explicitly
data_context = pj.load_data(
    "generic_data.csv",
    format_type="generic",
    capture_columns=["capture_2016", "capture_2017", "capture_2018"]
)
```

## Data Loading API

### Basic Loading

```python
# Automatic format detection (recommended)
data_context = pj.load_data("data.csv")

# Explicit format specification
data_context = pj.load_data("data.csv", format_type="rmark")
data_context = pj.load_data("data.csv", format_type="y_column")
data_context = pj.load_data("data.csv", format_type="generic")
```

### Advanced Loading Options

```python
# Load with custom parameters
data_context = pj.load_data(
    "data.csv",
    format_type="generic",
    capture_columns=["Y2016", "Y2017", "Y2018"],
    id_column="individual_id",
    covariate_columns=["sex", "age", "region"],
    na_values=["", "NA", "NULL", -9999],
    encoding="utf-8"
)
```

### Loading from Different Sources

```python
# From file path (string or Path object)
data_context = pj.load_data("/path/to/data.csv")
data_context = pj.load_data(Path("data.csv"))

# From pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
data_context = pj.load_data(df)

# From URL
data_context = pj.load_data("https://example.com/data.csv")
```

## Data Validation

Pradel-JAX performs comprehensive data validation to catch common issues early.

### Automatic Validation

```python
data_context = pj.load_data("data.csv")

# Validation is performed automatically during loading
if data_context.is_valid:
    print("Data is valid!")
else:
    print("Validation errors found:")
    for error in data_context.validation_errors:
        print(f"  - {error}")
```

### Manual Validation

```python
# Run validation explicitly
validation_result = data_context.validate()

print(f"Valid: {validation_result.is_valid}")
print(f"Warnings: {len(validation_result.warnings)}")
print(f"Errors: {len(validation_result.errors)}")

# Detailed validation report
print(validation_result.summary())
```

### Validation Checks

The framework performs these validation checks:

1. **Capture History Validation**
   - All entries are 0 or 1
   - No missing values in capture data
   - At least one capture per individual
   - Consistent capture history length

2. **Covariate Validation**
   - Consistent covariate values across time-varying variables
   - No excessive missing values
   - Appropriate data types

3. **Data Structure Validation**
   - Consistent number of occasions
   - No duplicate individuals (if ID column present)
   - Proper temporal ordering

4. **Statistical Validation**
   - Sufficient data for model fitting
   - Reasonable capture probabilities
   - No degenerate cases

### Handling Validation Issues

```python
# Common validation fixes
if not data_context.is_valid:
    # Remove individuals with no captures
    data_context = data_context.remove_never_captured()
    
    # Handle missing covariates
    data_context = data_context.fill_missing_covariates(method="mode")
    
    # Remove problematic individuals
    data_context = data_context.remove_individuals(condition="invalid_captures")
```

## Working with Large Datasets

### Memory-Efficient Loading

```python
# Load large datasets with chunking
data_context = pj.load_data(
    "large_dataset.csv",
    chunksize=10000,  # Process 10k rows at a time
    low_memory=True   # Optimize memory usage
)
```

### Sampling Large Datasets

```python
# Load with stratified sampling
data_context = pj.load_data_with_sampling(
    "large_dataset.csv",
    sample_size=1000,
    stratify_by="sex",  # Maintain sex distribution
    random_state=42
)

# Stratified sampling by multiple variables
data_context = pj.load_data_with_sampling(
    "large_dataset.csv",
    sample_size=5000,
    stratify_by=["sex", "age_class"],
    weights="capture_frequency"  # Weight by capture frequency
)
```

### Performance Optimization

```python
# Optimize data types for memory efficiency
data_context = pj.load_data(
    "data.csv",
    optimize_memory=True,  # Convert to efficient data types
    compress=True          # Compress in memory
)

# Parallel processing for very large datasets
data_context = pj.load_data(
    "huge_dataset.csv",
    n_jobs=4,  # Use 4 cores for processing
    batch_size=50000
)
```

## Time-Varying Covariates

Pradel-JAX fully supports time-varying covariates, which change across capture occasions.

### Structure Requirements

Time-varying covariates should follow naming conventions:
- `covariate_YEAR` format (e.g., `age_2016`, `age_2017`)
- `covariate_OCCASION` format (e.g., `weight_1`, `weight_2`)

### Loading Time-Varying Data

```python
# Automatic detection of time-varying covariates
data_context = pj.load_data("time_varying_data.csv")

# Check what was detected
print("Time-varying covariates:")
for covariate, occasions in data_context.time_varying_covariates.items():
    print(f"  {covariate}: {occasions} occasions")

# Example data structure:
# age_2016, age_2017, age_2018, tier_2016, tier_2017, tier_2018, sex
# 1,        2,        3,        1,         1,         2,         M
```

### Working with Time-Varying Covariates

```python
# Access time-varying covariate matrices
age_matrix = data_context.get_time_varying_covariate("age")
print(f"Age matrix shape: {age_matrix.shape}")  # (n_individuals, n_occasions)

# Validate temporal consistency
validation = data_context.validate_time_varying_covariates()
if validation.has_issues:
    print("Time-varying covariate issues:")
    for issue in validation.issues:
        print(f"  - {issue}")
```

### Example: Age and Tier Time-Varying

```python
# Load data with age and tier varying over time
data_context = pj.load_data("nebraska_data.csv")

# Examine the time-varying structure
print("Data structure:")
print(f"  Individuals: {data_context.n_individuals}")
print(f"  Occasions: {data_context.n_occasions}")
print(f"  Time-varying covariates: {list(data_context.time_varying_covariates.keys())}")

# Age progression validation
age_matrix = data_context.get_time_varying_covariate("age")
age_changes = age_matrix[:, 1:] - age_matrix[:, :-1]
print(f"Age changes per year: mean={age_changes.mean():.2f}, std={age_changes.std():.2f}")
```

## Data Sampling and Subsets

### Creating Training/Validation Splits

```python
# Simple random split
train_data, val_data = pj.train_validation_split(
    data_context, 
    test_size=0.2,
    random_state=42
)

# Stratified split maintaining covariate distributions
train_data, val_data = pj.train_validation_split(
    data_context,
    test_size=0.2,
    stratify_by="sex",
    random_state=42
)
```

### Stratified Sampling

```python
# Sample maintaining population structure
sample_data = pj.stratified_sample(
    data_context,
    n_samples=1000,
    stratify_columns=["sex", "age_class"],
    proportional=True  # Maintain original proportions
)

# Custom stratification weights
sample_data = pj.stratified_sample(
    data_context,
    n_samples=1000,
    stratify_columns=["region"],
    weights={"north": 0.6, "south": 0.4}
)
```

### Subset Selection

```python
# Filter by covariate values
male_data = data_context.filter(sex="M")
adult_data = data_context.filter(age_class="adult")

# Complex filtering
subset_data = data_context.filter(
    sex="M", 
    age_class="adult",
    capture_count=lambda x: x >= 3  # At least 3 captures
)

# Time-based filtering
recent_data = data_context.filter_occasions(start=2018, end=2022)
```

## Data Context Properties

The `DataContext` object provides rich information about your data:

```python
# Basic properties
print(f"Individuals: {data_context.n_individuals}")
print(f"Occasions: {data_context.n_occasions}")
print(f"Parameters needed: {data_context.n_parameters}")

# Capture summary
print(f"Total captures: {data_context.total_captures}")
print(f"Capture rate: {data_context.capture_rate:.3f}")
print(f"Individuals never captured: {data_context.never_captured.sum()}")

# Covariate summary
print("\nCovariates:")
for name, info in data_context.covariate_info.items():
    print(f"  {name}: {info['type']} ({info['unique_values']} unique values)")

# Temporal information
if data_context.has_time_varying_covariates:
    print(f"\nTime-varying covariates: {list(data_context.time_varying_covariates.keys())}")
```

## Troubleshooting

### Common Data Loading Issues

1. **Format Not Detected**
   ```python
   # Explicitly specify format
   data_context = pj.load_data("data.csv", format_type="rmark")
   ```

2. **Capture History Parsing Errors**
   ```python
   # Handle different separators or formats
   data_context = pj.load_data(
       "data.csv",
       capture_separator=",",  # For comma-separated histories
       preserve_leading_zeros=True  # Keep "01010" instead of "1010"
   )
   ```

3. **Covariate Type Issues**
   ```python
   # Specify data types explicitly
   data_context = pj.load_data(
       "data.csv",
       dtype={"sex": "category", "age": "float32", "weight": "float64"}
   )
   ```

4. **Missing Value Handling**
   ```python
   # Custom missing value treatment
   data_context = pj.load_data(
       "data.csv",
       na_values=["", "NA", "NULL", -999],
       fill_missing="mode",  # or "mean", "median", "drop"
   )
   ```

5. **Memory Issues with Large Datasets**
   ```python
   # Use chunking for very large datasets
   data_context = pj.load_data(
       "huge_dataset.csv",
       chunksize=10000,
       low_memory=True
   )
   ```

### Data Quality Issues

1. **Individuals with No Captures**
   ```python
   # Remove or flag individuals never captured
   clean_data = data_context.remove_never_captured()
   # Or identify them for investigation
   never_captured = data_context.get_never_captured_individuals()
   ```

2. **Inconsistent Capture Histories**
   ```python
   # Validate and fix capture history issues
   validation = data_context.validate_capture_histories()
   if not validation.is_valid:
       fixed_data = data_context.fix_capture_histories()
   ```

3. **Time-Varying Covariate Issues**
   ```python
   # Check for temporal consistency
   tv_validation = data_context.validate_time_varying_covariates()
   if tv_validation.has_warnings:
       print("Potential issues:")
       for warning in tv_validation.warnings:
           print(f"  - {warning}")
   ```

### Performance Tips

1. **Use appropriate data types** - Convert strings to categories, use appropriate numeric types
2. **Sample large datasets** - Use stratified sampling for development and testing
3. **Validate data early** - Catch issues before expensive model fitting
4. **Cache processed data** - Save DataContext objects for reuse
5. **Monitor memory usage** - Use memory profiling for very large datasets

### Getting Help

If you encounter data loading issues:

1. **Check the data format** - Ensure your data matches one of the supported formats
2. **Validate data quality** - Run validation checks to identify specific issues
3. **Consult examples** - See `examples/` directory for format-specific examples
4. **Report issues** - Open GitHub issues for bugs or feature requests

---

**Next Steps:**
- [Model Specification Guide](model-specification.md) - Learn how to specify and configure models
- [Formula System Guide](formulas.md) - Master the R-style formula syntax
- [Optimization Guide](optimization.md) - Understand optimization strategies and tuning