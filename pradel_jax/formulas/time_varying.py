"""
Time-varying covariate support for pradel-jax design matrices.

Provides robust handling of time-varying covariates with proper statistical foundations.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import re

from ..core.exceptions import ModelSpecificationError, DataFormatError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TimeVaryingCovariateInfo:
    """Information about time-varying covariates."""
    base_name: str
    time_occasions: List[str]  # e.g., ['2016', '2017', '2018']
    data_matrix: np.ndarray  # shape: (n_individuals, n_occasions)
    is_categorical: bool = False
    categories: Optional[List[str]] = None
    
    @property
    def n_occasions(self) -> int:
        return len(self.time_occasions)
    
    @property
    def n_individuals(self) -> int:
        return self.data_matrix.shape[0]


class TimeVaryingCovariateDetector:
    """
    Detects and extracts time-varying covariates from data.
    
    Supports common patterns:
    - age_2016, age_2017, age_2018 (underscore pattern)
    - Y2016, Y2017, Y2018 (prefix pattern)
    - variable.2016, variable.2017 (dot pattern)
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def detect_time_varying_patterns(self, data: pd.DataFrame) -> Dict[str, TimeVaryingCovariateInfo]:
        """
        Detect time-varying covariate patterns in data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary mapping base variable names to TimeVaryingCovariateInfo
        """
        time_varying_covariates = {}
        
        # Pattern 1: underscore pattern (age_2016, age_2017, ...)
        underscore_groups = self._detect_underscore_pattern(data)
        time_varying_covariates.update(underscore_groups)
        
        # Pattern 2: prefix pattern (Y2016, Y2017, ...)
        prefix_groups = self._detect_prefix_pattern(data)
        time_varying_covariates.update(prefix_groups)
        
        # Pattern 3: dot pattern (variable.2016, variable.2017, ...)
        dot_groups = self._detect_dot_pattern(data)
        time_varying_covariates.update(dot_groups)
        
        self.logger.info(f"Detected {len(time_varying_covariates)} time-varying covariate groups")
        
        return time_varying_covariates
    
    def _detect_underscore_pattern(self, data: pd.DataFrame) -> Dict[str, TimeVaryingCovariateInfo]:
        """Detect pattern: base_name_year (e.g., age_2016, age_2017)."""
        groups = {}
        
        # Find columns matching pattern: name_year
        pattern = re.compile(r'^(.+)_(\d{4})$')
        
        base_names = {}
        for col in data.columns:
            match = pattern.match(col)
            if match:
                base_name, year = match.groups()
                if base_name not in base_names:
                    base_names[base_name] = []
                base_names[base_name].append((year, col))
        
        # Create TimeVaryingCovariateInfo for groups with multiple years
        for base_name, year_cols in base_names.items():
            if len(year_cols) >= 2:  # Need at least 2 time points
                # Sort by year
                year_cols.sort(key=lambda x: x[0])
                
                years = [year for year, col in year_cols]
                cols = [col for year, col in year_cols]
                
                # Extract data matrix
                data_matrix = data[cols].values
                
                # Check if categorical
                is_categorical = self._is_categorical_data(data[cols])
                categories = None
                if is_categorical:
                    # Get unique values across all time points
                    all_values = pd.concat([data[col] for col in cols]).dropna().unique()
                    categories = sorted(all_values.tolist())
                
                groups[base_name] = TimeVaryingCovariateInfo(
                    base_name=base_name,
                    time_occasions=years,
                    data_matrix=data_matrix,
                    is_categorical=is_categorical,
                    categories=categories
                )
                
                self.logger.debug(f"Detected time-varying: {base_name} with {len(years)} occasions")
        
        return groups
    
    def _detect_prefix_pattern(self, data: pd.DataFrame) -> Dict[str, TimeVaryingCovariateInfo]:
        """Detect pattern: prefix_year (e.g., Y2016, Y2017)."""
        groups = {}
        
        # Find columns matching pattern: letter(s) + year
        pattern = re.compile(r'^([A-Za-z]+)(\d{4})$')
        
        base_names = {}
        for col in data.columns:
            match = pattern.match(col)
            if match:
                prefix, year = match.groups()
                if prefix not in base_names:
                    base_names[prefix] = []
                base_names[prefix].append((year, col))
        
        # Create groups for prefixes with multiple years
        for prefix, year_cols in base_names.items():
            if len(year_cols) >= 2:
                year_cols.sort(key=lambda x: x[0])
                
                years = [year for year, col in year_cols]
                cols = [col for year, col in year_cols]
                
                data_matrix = data[cols].values
                
                is_categorical = self._is_categorical_data(data[cols])
                categories = None
                if is_categorical:
                    all_values = pd.concat([data[col] for col in cols]).dropna().unique()
                    categories = sorted(all_values.tolist())
                
                groups[prefix] = TimeVaryingCovariateInfo(
                    base_name=prefix,
                    time_occasions=years,
                    data_matrix=data_matrix,
                    is_categorical=is_categorical,
                    categories=categories
                )
        
        return groups
    
    def _detect_dot_pattern(self, data: pd.DataFrame) -> Dict[str, TimeVaryingCovariateInfo]:
        """Detect pattern: base.year (e.g., variable.2016, variable.2017)."""
        groups = {}
        
        pattern = re.compile(r'^(.+)\.(\d{4})$')
        
        base_names = {}
        for col in data.columns:
            match = pattern.match(col)
            if match:
                base_name, year = match.groups()
                if base_name not in base_names:
                    base_names[base_name] = []
                base_names[base_name].append((year, col))
        
        for base_name, year_cols in base_names.items():
            if len(year_cols) >= 2:
                year_cols.sort(key=lambda x: x[0])
                
                years = [year for year, col in year_cols]
                cols = [col for year, col in year_cols]
                
                data_matrix = data[cols].values
                
                is_categorical = self._is_categorical_data(data[cols])
                categories = None
                if is_categorical:
                    all_values = pd.concat([data[col] for col in cols]).dropna().unique()
                    categories = sorted(all_values.tolist())
                
                groups[base_name] = TimeVaryingCovariateInfo(
                    base_name=base_name,
                    time_occasions=years,
                    data_matrix=data_matrix,
                    is_categorical=is_categorical,
                    categories=categories
                )
        
        return groups
    
    def _is_categorical_data(self, data_subset: pd.DataFrame) -> bool:
        """
        Determine if data should be treated as categorical.
        
        Uses statistical criteria to determine categorical nature.
        """
        # Check data types
        for col in data_subset.columns:
            if data_subset[col].dtype == 'object':
                return True
        
        # Check for integer data with limited unique values
        for col in data_subset.columns:
            unique_vals = data_subset[col].dropna().unique()
            
            # If all integer values and < 10 unique values, likely categorical
            if len(unique_vals) < 10:
                # Check if all values are integers (could be coded categoricals)
                try:
                    int_vals = [int(v) for v in unique_vals if not pd.isna(v)]
                    float_vals = [float(v) for v in unique_vals if not pd.isna(v)]
                    if int_vals == float_vals:  # All are integers
                        return True
                except (ValueError, TypeError):
                    pass
        
        return False


class TimeVaryingDesignMatrixBuilder:
    """
    Builds design matrices incorporating time-varying covariates.
    
    Handles the statistical challenges of time-varying covariates:
    - Proper indexing for each time occasion
    - Handling missing values across time
    - Appropriate contrasts for categorical variables
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.detector = TimeVaryingCovariateDetector()
    
    def build_time_varying_matrix(
        self,
        covariate_name: str,
        time_varying_info: TimeVaryingCovariateInfo,
        n_occasions: int,
        parameter_occasions: List[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build design matrix for time-varying covariate.
        
        Args:
            covariate_name: Name of the covariate
            time_varying_info: Information about the time-varying covariate
            n_occasions: Total number of modeling occasions
            parameter_occasions: Which occasions this parameter applies to
            
        Returns:
            Tuple of (design_matrix, column_names)
        """
        if parameter_occasions is None:
            parameter_occasions = list(range(n_occasions))
        
        n_individuals = time_varying_info.n_individuals
        
        if time_varying_info.is_categorical:
            return self._build_categorical_time_varying_matrix(
                covariate_name, time_varying_info, n_individuals, parameter_occasions
            )
        else:
            return self._build_numeric_time_varying_matrix(
                covariate_name, time_varying_info, n_individuals, parameter_occasions
            )
    
    def _build_numeric_time_varying_matrix(
        self,
        covariate_name: str,
        time_varying_info: TimeVaryingCovariateInfo,
        n_individuals: int,
        parameter_occasions: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Build matrix for numeric time-varying covariate."""
        
        # For Pradel models, we need to handle the fact that parameters
        # apply to intervals between occasions, not occasions themselves
        
        data_matrix = time_varying_info.data_matrix
        n_data_occasions = data_matrix.shape[1]
        
        # Create design matrix - individuals are rows
        # For time-varying covariates, we typically use the value at the
        # beginning of each interval (or average over interval)
        
        design_columns = []
        column_names = []
        
        for occasion in parameter_occasions:
            if occasion < n_data_occasions:
                # Use value at this occasion
                column_data = data_matrix[:, occasion].astype(np.float32)
            elif occasion > 0 and (occasion - 1) < n_data_occasions:
                # Use previous occasion if current not available
                column_data = data_matrix[:, occasion - 1].astype(np.float32)
            else:
                # Use last available occasion
                column_data = data_matrix[:, -1].astype(np.float32)
            
            # Handle missing values
            if np.any(np.isnan(column_data)):
                # Fill with individual mean or overall mean
                individual_means = np.nanmean(data_matrix, axis=1)
                for i in range(len(column_data)):
                    if np.isnan(column_data[i]):
                        if not np.isnan(individual_means[i]):
                            column_data[i] = individual_means[i]
                        else:
                            column_data[i] = np.nanmean(data_matrix)
            
            design_columns.append(column_data)
            column_names.append(f"{covariate_name}_t{occasion}")
        
        # For simple time-constant effect, we might want just one column
        # that uses appropriate time-varying values
        if len(parameter_occasions) == 1:
            return np.column_stack(design_columns), column_names
        else:
            # Multiple occasions - could return separate columns or
            # a single column with appropriate time values
            return np.column_stack(design_columns), column_names
    
    def _build_categorical_time_varying_matrix(
        self,
        covariate_name: str,
        time_varying_info: TimeVaryingCovariateInfo,
        n_individuals: int,
        parameter_occasions: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Build matrix for categorical time-varying covariate."""
        
        data_matrix = time_varying_info.data_matrix
        categories = time_varying_info.categories
        n_data_occasions = data_matrix.shape[1]
        
        if len(categories) <= 1:
            # Only one category - return constant column
            column = np.ones(n_individuals, dtype=np.float32)
            return np.column_stack([column]), [covariate_name]
        
        # Create dummy variables for each category (drop first for identifiability)
        design_columns = []
        column_names = []
        
        for cat_idx, category in enumerate(categories[1:], 1):  # Skip first category
            for occasion in parameter_occasions:
                if occasion < n_data_occasions:
                    occasion_data = data_matrix[:, occasion]
                elif occasion > 0 and (occasion - 1) < n_data_occasions:
                    occasion_data = data_matrix[:, occasion - 1]
                else:
                    occasion_data = data_matrix[:, -1]
                
                # Create dummy variable
                dummy_column = (occasion_data == cat_idx).astype(np.float32)
                
                design_columns.append(dummy_column)
                column_names.append(f"{covariate_name}_{category}_t{occasion}")
        
        return np.column_stack(design_columns), column_names
    
    def expand_data_context_with_time_varying(
        self,
        data_context: Any,  # DataContext
        data: pd.DataFrame
    ) -> Any:  # Updated DataContext
        """
        Expand DataContext to include proper time-varying covariate support.
        
        Args:
            data_context: Original DataContext
            data: Original DataFrame (needed for pattern detection)
            
        Returns:
            Updated DataContext with time-varying covariate information
        """
        # Detect time-varying patterns
        time_varying_covariates = self.detector.detect_time_varying_patterns(data)
        
        if not time_varying_covariates:
            self.logger.info("No time-varying covariates detected")
            return data_context
        
        # Add time-varying information to covariates
        updated_covariates = data_context.covariates.copy()
        updated_covariate_info = data_context.covariate_info.copy()
        
        for base_name, tv_info in time_varying_covariates.items():
            # Add the time-varying data matrix
            updated_covariates[f"{base_name}_time_varying"] = jnp.array(tv_info.data_matrix)
            
            # Add metadata
            updated_covariates[f"{base_name}_is_time_varying"] = True
            updated_covariates[f"{base_name}_time_occasions"] = tv_info.time_occasions
            
            if tv_info.is_categorical:
                updated_covariates[f"{base_name}_categories"] = tv_info.categories
                updated_covariates[f"{base_name}_is_categorical"] = True
            
            # Create updated covariate info
            from ..data.adapters import CovariateInfo
            updated_covariate_info[base_name] = CovariateInfo(
                name=base_name,
                dtype="time_varying_categorical" if tv_info.is_categorical else "time_varying_numeric",
                is_time_varying=True,
                is_categorical=tv_info.is_categorical,
                levels=tv_info.categories,
                time_occasions=tv_info.time_occasions
            )
        
        self.logger.info(f"Added {len(time_varying_covariates)} time-varying covariates to context")
        
        # Create new DataContext with updated information
        return type(data_context)(
            capture_matrix=data_context.capture_matrix,
            covariates=updated_covariates,
            covariate_info=updated_covariate_info,
            n_individuals=data_context.n_individuals,
            n_occasions=data_context.n_occasions,
            occasion_names=data_context.occasion_names,
            individual_ids=data_context.individual_ids,
            metadata={**(data_context.metadata or {}), 'has_time_varying_covariates': True}
        )


def detect_and_process_time_varying_covariates(
    data: pd.DataFrame,
    data_context: Any
) -> Tuple[Any, Dict[str, TimeVaryingCovariateInfo]]:
    """
    Convenience function to detect and process time-varying covariates.
    
    Args:
        data: Original DataFrame
        data_context: Original DataContext
        
    Returns:
        Tuple of (updated_data_context, time_varying_info_dict)
    """
    builder = TimeVaryingDesignMatrixBuilder()
    
    # Detect time-varying patterns
    time_varying_info = builder.detector.detect_time_varying_patterns(data)
    
    # Update data context
    updated_context = builder.expand_data_context_with_time_varying(data_context, data)
    
    return updated_context, time_varying_info