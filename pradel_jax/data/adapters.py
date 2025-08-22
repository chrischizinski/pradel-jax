"""
Data format adapters for pradel-jax.

Provides extensible system for handling different capture-recapture data formats.
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

from ..core.exceptions import DataFormatError
from ..utils.logging import get_logger
from ..utils.validation import validate_capture_matrix


logger = get_logger(__name__)


@dataclass
class CovariateInfo:
    """Information about a covariate."""
    name: str
    dtype: str
    is_time_varying: bool = False
    is_categorical: bool = False
    levels: Optional[List[str]] = None
    time_occasions: Optional[List[str]] = None


@dataclass 
class DataContext:
    """Context object containing processed data and metadata."""
    capture_matrix: jnp.ndarray
    covariates: Dict[str, jnp.ndarray]
    covariate_info: Dict[str, CovariateInfo]
    n_individuals: int
    n_occasions: int
    occasion_names: Optional[List[str]] = None
    individual_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize DataContext to a pickle-safe dictionary.
        
        Converts JAX arrays to numpy arrays for cross-process serialization.
        """
        # Convert JAX arrays to numpy arrays
        capture_matrix_np = np.array(self.capture_matrix)
        
        covariates_np = {}
        for name, value in self.covariates.items():
            if isinstance(value, jnp.ndarray):
                covariates_np[name] = np.array(value)
            else:
                # Keep non-array values (lists, bools, etc.) as-is
                covariates_np[name] = value
        
        # Convert CovariateInfo objects to dicts
        covariate_info_dict = {}
        for name, info in self.covariate_info.items():
            covariate_info_dict[name] = {
                'name': info.name,
                'dtype': info.dtype,
                'is_time_varying': info.is_time_varying,
                'is_categorical': info.is_categorical,
                'levels': info.levels,
                'time_occasions': info.time_occasions
            }
        
        return {
            'capture_matrix': capture_matrix_np,
            'covariates': covariates_np,
            'covariate_info': covariate_info_dict,
            'n_individuals': self.n_individuals,
            'n_occasions': self.n_occasions,
            'occasion_names': self.occasion_names,
            'individual_ids': self.individual_ids,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'DataContext':
        """
        Deserialize DataContext from dictionary.
        
        Converts numpy arrays back to JAX arrays and reconstructs objects.
        """
        # Convert numpy arrays back to JAX arrays
        capture_matrix = jnp.array(data_dict['capture_matrix'])
        
        covariates = {}
        for name, value in data_dict['covariates'].items():
            if isinstance(value, np.ndarray):
                covariates[name] = jnp.array(value)
            else:
                # Keep non-array values as-is
                covariates[name] = value
        
        # Reconstruct CovariateInfo objects
        covariate_info = {}
        for name, info_dict in data_dict['covariate_info'].items():
            covariate_info[name] = CovariateInfo(**info_dict)
        
        return cls(
            capture_matrix=capture_matrix,
            covariates=covariates,
            covariate_info=covariate_info,
            n_individuals=data_dict['n_individuals'],
            n_occasions=data_dict['n_occasions'],
            occasion_names=data_dict['occasion_names'],
            individual_ids=data_dict['individual_ids'],
            metadata=data_dict['metadata']
        )


class DataFormatAdapter(ABC):
    """Abstract base class for data format adapters."""
    
    @abstractmethod
    def detect_format(self, data: pd.DataFrame) -> bool:
        """
        Detect if this adapter can handle the data format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if this adapter can handle the format
        """
        pass
    
    @abstractmethod
    def extract_capture_histories(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract capture matrix from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Capture matrix (individuals x occasions)
        """
        pass
    
    @abstractmethod
    def extract_covariates(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract covariates from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of covariate arrays
        """
        pass
    
    @abstractmethod
    def get_covariate_info(self, data: pd.DataFrame) -> Dict[str, CovariateInfo]:
        """
        Get information about covariates.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of covariate information
        """
        pass
    
    def process(self, data: pd.DataFrame) -> DataContext:
        """
        Process data into DataContext.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataContext
        """
        logger.info(f"Processing data with {self.__class__.__name__}")
        
        # Extract components
        capture_matrix = self.extract_capture_histories(data)
        covariates = self.extract_covariates(data)
        covariate_info = self.get_covariate_info(data)
        
        # Validate capture matrix
        validate_capture_matrix(capture_matrix)
        
        # Convert to JAX arrays
        capture_matrix_jax = jnp.array(capture_matrix)
        covariates_jax = {}
        
        for name, array in covariates.items():
            # Skip metadata fields that can't be converted to JAX arrays
            if name.endswith('_categories') or name.endswith('_is_categorical'):
                covariates_jax[name] = array  # Keep as-is (list or bool)
            else:
                covariates_jax[name] = jnp.array(array)  # Convert to JAX array
        
        n_individuals, n_occasions = capture_matrix.shape
        
        # Get individual IDs if available
        individual_ids = None
        if 'individual_id' in data.columns:
            individual_ids = data['individual_id'].tolist()
        elif 'id' in data.columns:
            individual_ids = data['id'].tolist()
        
        logger.info(
            f"Processed data: {n_individuals} individuals, {n_occasions} occasions, "
            f"{len(covariates)} covariates"
        )
        
        return DataContext(
            capture_matrix=capture_matrix_jax,
            covariates=covariates_jax,
            covariate_info=covariate_info,
            n_individuals=n_individuals,
            n_occasions=n_occasions,
            individual_ids=individual_ids,
            metadata={'adapter': self.__class__.__name__}
        )


class RMarkFormatAdapter(DataFormatAdapter):
    """Adapter for RMark-style data format (ch column + covariates)."""
    
    def detect_format(self, data: pd.DataFrame) -> bool:
        """Detect RMark format: has 'ch' column with capture histories."""
        return 'ch' in data.columns
    
    def extract_capture_histories(self, data: pd.DataFrame) -> np.ndarray:
        """Extract capture histories from 'ch' column."""
        if 'ch' not in data.columns:
            raise DataFormatError(
                specific_issue="Missing 'ch' column for capture histories",
                suggestions=[
                    "RMark format requires a 'ch' column",
                    "Check column names in your data",
                    "Ensure capture histories are in the correct column",
                ]
            )
        
        capture_histories = data['ch'].astype(str)
        
        # Validate format
        if not capture_histories.str.match(r'^[01]+$').all():
            invalid_count = (~capture_histories.str.match(r'^[01]+$')).sum()
            raise DataFormatError(
                specific_issue=f"{invalid_count} capture histories contain invalid characters",
                suggestions=[
                    "Capture histories must contain only '0' and '1' characters",
                    "Check for missing values or special characters",
                    "Example valid format: '0110100'",
                ]
            )
        
        # Check consistent lengths
        lengths = capture_histories.str.len()
        if lengths.nunique() > 1:
            raise DataFormatError(
                specific_issue=f"Inconsistent capture history lengths: {sorted(lengths.unique())}",
                suggestions=[
                    "All capture histories must have the same length",
                    "Pad shorter histories with leading zeros if needed",
                    "Check for data truncation or formatting issues",
                ]
            )
        
        # Convert to matrix
        n_occasions = lengths.iloc[0]
        capture_matrix = np.zeros((len(data), n_occasions), dtype=np.int32)
        
        for i, ch in enumerate(capture_histories):
            capture_matrix[i, :] = [int(c) for c in ch]
        
        return capture_matrix
    
    def extract_covariates(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract covariates (all columns except 'ch' and ID columns)."""
        # Identify covariate columns
        exclude_cols = {'ch', 'individual_id', 'id', 'person_id'}
        covariate_cols = [col for col in data.columns if col not in exclude_cols]
        
        covariates = {}
        
        for col in covariate_cols:
            if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
                # Store categorical variable as-is, design matrix builder will handle dummy variables
                # Convert to numeric codes for easier processing
                categorical_data = pd.Categorical(data[col])
                covariates[col] = categorical_data.codes.astype(np.float32)
                # Store category information for design matrix building
                covariates[f'{col}_categories'] = categorical_data.categories.tolist()
                covariates[f'{col}_is_categorical'] = True
            else:
                # Numeric variable
                values = data[col].values.astype(np.float32)
                # Handle NaN values
                if np.any(np.isnan(values)):
                    logger.warning(f"Column '{col}' contains NaN values - filling with mean")
                    values = np.nan_to_num(values, nan=np.nanmean(values))
                covariates[col] = values
        
        return covariates
    
    def get_covariate_info(self, data: pd.DataFrame) -> Dict[str, CovariateInfo]:
        """Get information about covariates."""
        exclude_cols = {'ch', 'individual_id', 'id', 'person_id'}
        covariate_cols = [col for col in data.columns if col not in exclude_cols]
        
        covariate_info = {}
        
        for col in covariate_cols:
            is_categorical = (data[col].dtype == 'object' or 
                            str(data[col].dtype) == 'category')
            
            if is_categorical:
                # Create info for dummy variables
                levels = data[col].unique().tolist()
                dummies = pd.get_dummies(data[col], prefix=col)
                for dummy_col in dummies.columns:
                    # Extract level name safely
                    level_name = dummy_col.split('_', 1)[1] if '_' in dummy_col else dummy_col
                    covariate_info[dummy_col] = CovariateInfo(
                        name=dummy_col,
                        dtype='binary',
                        is_categorical=True,
                        levels=[level_name]
                    )
            else:
                covariate_info[col] = CovariateInfo(
                    name=col,
                    dtype=str(data[col].dtype),
                    is_categorical=False
                )
        
        return covariate_info


class GenericFormatAdapter(DataFormatAdapter):
    """
    Generic adapter that can handle various formats with configuration.
    
    Supports:
    - Y-column format (Y2016, Y2017, etc.)
    - Explicit capture matrices
    - Custom column specifications
    """
    
    def __init__(
        self,
        capture_columns: Optional[List[str]] = None,
        covariate_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None
    ):
        self.capture_columns = capture_columns
        self.covariate_columns = covariate_columns
        self.id_column = id_column
    
    def detect_format(self, data: pd.DataFrame) -> bool:
        """Detect if this is a generic format we can handle."""
        # Check for Y-column format
        y_cols = [col for col in data.columns if col.startswith('Y') and col[1:].isdigit()]
        if len(y_cols) >= 3:  # Need at least 3 occasions
            return True
        
        # Check for explicit capture columns
        if self.capture_columns:
            return all(col in data.columns for col in self.capture_columns)
        
        # Check for numeric columns that could be captures
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            return True
        
        return False
    
    def extract_capture_histories(self, data: pd.DataFrame) -> np.ndarray:
        """Extract capture histories from various column formats."""
        if self.capture_columns:
            # Use explicitly specified columns
            capture_cols = self.capture_columns
        else:
            # Auto-detect Y-columns
            y_cols = [col for col in data.columns if col.startswith('Y') and col[1:].isdigit()]
            if y_cols:
                capture_cols = sorted(y_cols)
            else:
                # Use numeric columns as fallback
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                exclude_cols = {self.id_column} if self.id_column else set()
                capture_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not capture_cols:
            raise DataFormatError(
                specific_issue="No capture history columns found",
                suggestions=[
                    "Specify capture_columns explicitly",
                    "Use Y-prefixed columns (Y2016, Y2017, etc.)",
                    "Ensure numeric columns contain capture data",
                ]
            )
        
        # Extract and convert to binary
        capture_data = data[capture_cols].values
        capture_matrix = (capture_data > 0).astype(np.int32)
        
        return capture_matrix
    
    def extract_covariates(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract covariates from remaining columns."""
        # Create a copy to avoid modifying original data
        data_processed = data.copy()
        
        # CRITICAL FIX: Convert numeric codes to meaningful categorical labels
        data_processed = self._preprocess_categorical_variables(data_processed)
        
        # Determine which columns to exclude
        if self.capture_columns:
            exclude_cols = set(self.capture_columns)
        else:
            y_cols = [col for col in data_processed.columns if col.startswith('Y') and col[1:].isdigit()]
            if y_cols:
                exclude_cols = set(y_cols)
            else:
                exclude_cols = set()
        
        if self.id_column:
            exclude_cols.add(self.id_column)
        
        # Get covariate columns
        if self.covariate_columns:
            covariate_cols = self.covariate_columns
        else:
            covariate_cols = [col for col in data_processed.columns if col not in exclude_cols]
        
        covariates = {}
        
        for col in covariate_cols:
            if data_processed[col].dtype == 'object' or str(data_processed[col].dtype) == 'category':
                # Store categorical variable as-is, design matrix builder will handle dummy variables
                # Convert to numeric codes for easier processing
                categorical_data = pd.Categorical(data_processed[col])
                covariates[col] = categorical_data.codes.astype(np.float32)
                # Store category information for design matrix building
                covariates[f'{col}_categories'] = categorical_data.categories.tolist()
                covariates[f'{col}_is_categorical'] = True
            else:
                # Numeric variable
                values = data_processed[col].values.astype(np.float32)
                if np.any(np.isnan(values)):
                    values = np.nan_to_num(values, nan=np.nanmean(values))
                covariates[col] = values
        
        return covariates
    
    def get_covariate_info(self, data: pd.DataFrame) -> Dict[str, CovariateInfo]:
        """Get covariate information."""
        # Similar logic to extract_covariates but for metadata
        if self.capture_columns:
            exclude_cols = set(self.capture_columns)
        else:
            y_cols = [col for col in data.columns if col.startswith('Y') and col[1:].isdigit()]
            exclude_cols = set(y_cols) if y_cols else set()
        
        if self.id_column:
            exclude_cols.add(self.id_column)
        
        covariate_cols = (self.covariate_columns if self.covariate_columns 
                         else [col for col in data.columns if col not in exclude_cols])
        
        covariate_info = {}
        
        for col in covariate_cols:
            is_categorical = (data[col].dtype == 'object' or 
                            str(data[col].dtype) == 'category')
            
            if is_categorical:
                levels = data[col].unique().tolist()
                dummies = pd.get_dummies(data[col], prefix=col)
                for dummy_col in dummies.columns:
                    level_name = dummy_col.split('_', 1)[1] if '_' in dummy_col else dummy_col
                    covariate_info[dummy_col] = CovariateInfo(
                        name=dummy_col,
                        dtype='binary',
                        is_categorical=True,
                        levels=[level_name]
                    )
            else:
                covariate_info[col] = CovariateInfo(
                    name=col,
                    dtype=str(data[col].dtype),
                    is_categorical=False
                )
        
        return covariate_info
    
    def _preprocess_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FIX: Convert numeric codes to meaningful categorical labels.
        
        This addresses the validation audit finding that categorical variables
        like 'gender' and 'tier_history' use numeric codes instead of meaningful labels.
        """
        data_fixed = data.copy()
        
        # Fix gender encoding: 1.0 = Male, 2.0 = Female
        if 'gender' in data_fixed.columns:
            if data_fixed['gender'].dtype in ['float64', 'int64', 'float32', 'int32']:
                logger.info("Converting gender from numeric codes to categorical labels")
                gender_mapping = {1.0: 'Male', 2.0: 'Female'}
                data_fixed['gender'] = data_fixed['gender'].fillna(1.0).map(gender_mapping)
                # Fill any unmapped values with 'Male' (default)
                data_fixed['gender'] = data_fixed['gender'].fillna('Male')
        
        # Fix tier_history encoding: Reduce excessive categories
        if 'tier_history' in data_fixed.columns:
            if data_fixed['tier_history'].dtype in ['float64', 'int64', 'float32', 'int32']:
                logger.info("Converting tier_history from numeric codes to simplified categories")
                # Simplify tier_history by grouping similar values
                # This reduces the 91 unique values to meaningful categories
                def simplify_tier_history(value):
                    if pd.isna(value):
                        return 'Unknown'
                    value = float(value)
                    if value <= 1:
                        return 'Tier_1'
                    elif value <= 2:
                        return 'Tier_2'
                    elif value <= 3:
                        return 'Tier_3'
                    else:
                        return 'Tier_Higher'
                
                data_fixed['tier_history'] = data_fixed['tier_history'].apply(simplify_tier_history)
        
        # Fix tier encoding if present
        if 'tier' in data_fixed.columns:
            if data_fixed['tier'].dtype in ['float64', 'int64', 'float32', 'int32']:
                logger.info("Converting tier from numeric codes to categorical labels")
                # Map tier values for permit choice analysis: 0=no permit, 1/2=permit tier choice
                def map_tier(value):
                    if pd.isna(value):
                        return 'No_Permit'
                    value = float(value)
                    if value == 0:
                        return 'No_Permit'
                    elif value == 1:
                        return 'Tier_1_Permit'
                    elif value == 2:
                        return 'Tier_2_Permit'
                    else:
                        # Handle any unexpected values gracefully
                        logger.warning(f"Unexpected tier value: {value}, mapping to No_Permit")
                        return 'No_Permit'
                
                data_fixed['tier'] = data_fixed['tier'].apply(map_tier)
                
                # Create additional binary variables for analysis
                # Binary: Has permit (1 or 2) vs No permit (0)
                data_fixed['has_permit'] = data_fixed['tier'].apply(
                    lambda x: 'Yes' if x in ['Tier_1_Permit', 'Tier_2_Permit'] else 'No'
                )
                
                # Conditional: Among permit holders, which tier? (1 vs 2)
                data_fixed['permit_tier_choice'] = data_fixed['tier'].apply(
                    lambda x: 'Tier_1' if x == 'Tier_1_Permit' 
                             else 'Tier_2' if x == 'Tier_2_Permit' 
                             else 'No_Permit'
                )
        
        # Standardize continuous variables to prevent scaling issues
        continuous_vars = ['age']
        for var in continuous_vars:
            if var in data_fixed.columns:
                if data_fixed[var].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Center and scale
                    mean_val = data_fixed[var].mean()
                    std_val = data_fixed[var].std()
                    if std_val > 0:
                        data_fixed[var] = (data_fixed[var] - mean_val) / std_val
                        logger.info(f"Standardized {var}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        return data_fixed


# Registry of available adapters
_adapters = [
    RMarkFormatAdapter(),
    GenericFormatAdapter(),
]

def register_adapter(adapter: DataFormatAdapter) -> None:
    """Register a new data format adapter."""
    _adapters.insert(0, adapter)  # New adapters get priority


def detect_data_format(data: pd.DataFrame) -> DataFormatAdapter:
    """
    Automatically detect the appropriate data format adapter.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Appropriate adapter instance
        
    Raises:
        DataFormatError: If no suitable adapter is found
    """
    for adapter in _adapters:
        if adapter.detect_format(data):
            logger.info(f"Detected format: {adapter.__class__.__name__}")
            return adapter
    
    raise DataFormatError(
        specific_issue="Unable to detect data format",
        suggestions=[
            "Supported formats:",
            "  - RMark format: 'ch' column with capture histories",
            "  - Y-column format: Y2016, Y2017, etc.",
            "  - Custom format: specify columns explicitly",
            "Check data structure and column names",
            "Use GenericFormatAdapter with explicit configuration",
        ]
    )


def load_data(
    file_path: Union[str, Path],
    adapter: Optional[DataFormatAdapter] = None,
    **kwargs
) -> DataContext:
    """
    Load data from file using appropriate format adapter.
    
    Args:
        file_path: Path to data file
        adapter: Specific adapter to use (auto-detected if None)
        **kwargs: Additional arguments for pandas.read_csv
        
    Returns:
        Processed DataContext
        
    Raises:
        DataFormatError: If file cannot be loaded or format not recognized
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DataFormatError(
            specific_issue=f"File not found: {file_path}",
            suggestions=[
                "Check file path and name",
                "Ensure file exists and is readable",
            ]
        )
    
    # Load data
    try:
        if file_path.suffix.lower() == '.csv':
            # For CSV files, check if there's a 'ch' column and read it as string
            # to preserve leading zeros in capture histories
            dtype_dict = kwargs.get('dtype', {})
            if 'ch' not in dtype_dict:
                # Peek at the file to see if it has a 'ch' column
                sample_df = pd.read_csv(file_path, nrows=5)
                if 'ch' in sample_df.columns:
                    dtype_dict['ch'] = str
                    kwargs['dtype'] = dtype_dict
            
            data = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path, **kwargs)
        else:
            raise DataFormatError(
                specific_issue=f"Unsupported file format: {file_path.suffix}",
                suggestions=[
                    "Supported formats: CSV, Excel",
                    "Convert file to CSV format",
                ]
            )
    except Exception as e:
        raise DataFormatError(
            specific_issue=f"Failed to load file: {e}",
            suggestions=[
                "Check file format and encoding",
                "Ensure file is not corrupted",
                "Try opening file in a text editor",
            ]
        )
    
    # Detect or use specified adapter
    if adapter is None:
        adapter = detect_data_format(data)
    
    # Process data
    return adapter.process(data)