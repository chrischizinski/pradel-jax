"""
Core export functionality for pradel-jax.
Provides comprehensive parameter export and results management.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging

from ..data.adapters import DataContext
from ..optimization.parallel import ParallelOptimizationResult, ParallelModelSpec
from ..models.base import ModelResult

logger = logging.getLogger(__name__)

class ResultsExporter:
    """
    Comprehensive results export functionality for pradel-jax models.
    
    Handles parameter export, model comparison, and result persistence
    with full metadata and statistical information.
    """
    
    def __init__(self, include_design_info: bool = True, decimal_precision: int = 6):
        """
        Initialize results exporter.
        
        Args:
            include_design_info: Whether to include design matrix parameter names
            decimal_precision: Number of decimal places for numeric values
        """
        self.include_design_info = include_design_info
        self.decimal_precision = decimal_precision
    
    def export_results(
        self,
        results: List[Union[ParallelOptimizationResult, ModelResult]],
        model_specs: Optional[List[ParallelModelSpec]] = None,
        data_context: Optional[DataContext] = None,
        export_file: Optional[Union[str, Path]] = None,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Export comprehensive results from fitted models.
        
        Args:
            results: List of optimization/model results
            model_specs: List of model specifications (if available)
            data_context: Data context used for fitting (if available)
            export_file: Optional file path to save results
            include_metadata: Whether to include detailed metadata
            
        Returns:
            DataFrame with comprehensive model results
        """
        logger.info(f"Exporting results for {len(results)} models")
        
        export_data = []
        
        for i, result in enumerate(results):
            model_spec = model_specs[i] if model_specs and i < len(model_specs) else None
            export_record = self._extract_result_data(result, model_spec, data_context, i+1, include_metadata)
            export_data.append(export_record)
        
        # Create DataFrame
        export_df = pd.DataFrame(export_data)
        
        # Round numeric columns
        numeric_columns = export_df.select_dtypes(include=[np.number]).columns
        export_df[numeric_columns] = export_df[numeric_columns].round(self.decimal_precision)
        
        # Save to file if requested
        if export_file:
            export_path = Path(export_file)
            export_df.to_csv(export_path, index=False)
            logger.info(f"Results exported to: {export_path}")
        
        return export_df
    
    def _extract_result_data(
        self, 
        result: Union[ParallelOptimizationResult, ModelResult],
        model_spec: Optional[ParallelModelSpec],
        data_context: Optional[DataContext],
        model_id: int,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Extract data from a single result object."""
        
        # Basic information
        export_record = {
            'model_id': model_id,
            'model_name': getattr(result, 'model_name', f'model_{model_id}'),
            'converged': getattr(result, 'success', False)
        }
        
        # Handle failed models
        if not export_record['converged']:
            export_record['error_message'] = getattr(result, 'error_message', 'Unknown error')
            return export_record
        
        # Core model statistics (following MARK conventions)
        if hasattr(result, 'log_likelihood') and result.log_likelihood is not None:
            export_record['log_likelihood'] = result.log_likelihood
            export_record['minus_log_likelihood'] = -result.log_likelihood
        
        if hasattr(result, 'aic') and result.aic is not None:
            export_record['aic'] = result.aic
            
        if hasattr(result, 'bic') and result.bic is not None:
            export_record['bic'] = result.bic
            
        # MARK-style QAICc (quasi-AIC corrected) - placeholder for overdispersion handling
        if hasattr(result, 'aic') and result.aic is not None:
            # For now, QAICc = AIC (assuming no overdispersion). 
            # This should be enhanced to handle overdispersion parameter (c-hat)
            export_record['qaic'] = result.aic
            export_record['qaicc'] = result.aic  # Small sample correction would go here
        
        # Deviance (MARK reports deviance = -2 * log-likelihood)
        if hasattr(result, 'log_likelihood') and result.log_likelihood is not None:
            export_record['deviance'] = -2 * result.log_likelihood
        
        # Parameter information
        if hasattr(result, 'parameters') and result.parameters is not None:
            parameters = result.parameters
            export_record['n_parameters'] = len(parameters)
            
            # Add raw parameters by index
            for j, param_value in enumerate(parameters):
                export_record[f'param_{j+1}'] = param_value
        
        # Population dynamics (lambda estimates)
        self._add_lambda_estimates(export_record, result)
        
        # Formula information
        if model_spec and hasattr(model_spec, 'formula_spec'):
            export_record['formula_phi'] = model_spec.formula_spec.phi.formula_string
            export_record['formula_p'] = model_spec.formula_spec.p.formula_string  
            export_record['formula_f'] = model_spec.formula_spec.f.formula_string
        
        # Named parameters from design matrices
        if (self.include_design_info and model_spec and data_context and 
            hasattr(model_spec, 'formula_spec') and hasattr(result, 'parameters')):
            self._add_named_parameters(export_record, result, model_spec, data_context)
        
        # Optimization metadata
        if include_metadata:
            self._add_optimization_metadata(export_record, result)
        
        return export_record
    
    def _add_lambda_estimates(self, export_record: Dict[str, Any], result) -> None:
        """Add lambda (population growth rate) estimates to export record."""
        lambda_attrs = ['lambda_mean', 'lambda_std', 'lambda_median', 
                       'lambda_min', 'lambda_max', 'lambda_q25', 'lambda_q75']
        
        for attr in lambda_attrs:
            if hasattr(result, attr) and getattr(result, attr) is not None:
                export_record[attr] = getattr(result, attr)
    
    def _add_named_parameters(
        self, 
        export_record: Dict[str, Any], 
        result, 
        model_spec: ParallelModelSpec,
        data_context: DataContext
    ) -> None:
        """Add named parameters using design matrix information."""
        try:
            from ..models import PradelModel
            model = PradelModel()
            design_matrices = model.build_design_matrices(model_spec.formula_spec, data_context)
            
            # Extract parameter names from design matrices
            phi_names = [f'phi_{name}' for name in design_matrices['phi'].column_names]
            p_names = [f'p_{name}' for name in design_matrices['p'].column_names] 
            f_names = [f'f_{name}' for name in design_matrices['f'].column_names]
            
            all_param_names = phi_names + p_names + f_names
            
            # Add named parameters with estimates
            for param_name, param_value in zip(all_param_names, result.parameters):
                export_record[param_name] = param_value
                
                # Add MARK-style parameter export format
                export_record[f'{param_name}_estimate'] = param_value
                
                # Placeholder for standard errors (would come from Hessian/Fisher Information)
                # This should be computed from the optimization result
                if hasattr(result, 'parameter_se') and result.parameter_se is not None:
                    if len(result.parameter_se) > len(all_param_names):
                        se_value = result.parameter_se[all_param_names.index(param_name.split('_', 1)[1])]
                        export_record[f'{param_name}_se'] = se_value
                        
                        # MARK-style confidence intervals (estimate ± 1.96 * SE)
                        export_record[f'{param_name}_lcl'] = param_value - 1.96 * se_value
                        export_record[f'{param_name}_ucl'] = param_value + 1.96 * se_value
                
            # Store parameter structure info
            export_record['parameter_names'] = '; '.join(all_param_names)
            export_record['n_phi_params'] = len(phi_names)
            export_record['n_p_params'] = len(p_names)
            export_record['n_f_params'] = len(f_names)
            
            # MARK-style parameter structure summary
            param_structure = f"φ({len(phi_names)}) p({len(p_names)}) f({len(f_names)})"
            export_record['parameter_structure'] = param_structure
            
        except Exception as e:
            logger.warning(f"Could not extract named parameters: {e}")
            export_record['design_matrix_error'] = str(e)
    
    def _add_optimization_metadata(self, export_record: Dict[str, Any], result) -> None:
        """Add optimization-specific metadata."""
        metadata_attrs = [
            'strategy_used', 'fit_time', 'n_iterations', 
            'gradient_norm', 'random_seed', 'data_hash'
        ]
        
        for attr in metadata_attrs:
            if hasattr(result, attr) and getattr(result, attr) is not None:
                export_record[attr] = getattr(result, attr)
    
    def create_model_comparison_table(
        self, 
        export_df: pd.DataFrame,
        sort_by: str = 'aic',
        ascending: bool = True,
        use_qaic: bool = False
    ) -> pd.DataFrame:
        """
        Create a MARK-style model comparison table sorted by information criterion.
        
        Args:
            export_df: Export DataFrame from export_results()
            sort_by: Column to sort by ('aic', 'bic', 'qaic', 'qaicc')
            ascending: Whether to sort in ascending order
            use_qaic: Whether to use QAICc for overdispersion-adjusted comparison
            
        Returns:
            Sorted comparison table with delta values and weights (MARK format)
        """
        # Filter successful models
        successful_df = export_df[export_df['converged'] == True].copy()
        
        if len(successful_df) == 0:
            logger.warning("No successful models for comparison")
            return pd.DataFrame()
        
        # Use QAICc if requested and available, otherwise fall back to AIC
        if use_qaic and 'qaicc' in successful_df.columns:
            comparison_col = 'qaicc'
        else:
            comparison_col = sort_by
        
        # Sort by criterion
        if comparison_col in successful_df.columns:
            successful_df = successful_df.sort_values(comparison_col, ascending=ascending)
            
            # Calculate delta values and weights
            if comparison_col in ['aic', 'bic', 'qaic', 'qaicc']:
                min_ic = successful_df[comparison_col].min()
                delta_col = f'delta_{comparison_col}'
                weight_col = f'{comparison_col}_weight'
                
                successful_df[delta_col] = successful_df[comparison_col] - min_ic
                
                # Calculate Akaike weights (MARK standard)
                delta_ic = successful_df[delta_col]
                rel_likelihood = np.exp(-0.5 * delta_ic)
                weights = rel_likelihood / rel_likelihood.sum()
                successful_df[weight_col] = weights
                
                # MARK-style evidence ratios (weight of best model / weight of each model)
                best_weight = weights.iloc[0]
                successful_df['evidence_ratio'] = best_weight / weights
        
        # Add model ranking (MARK-style)
        successful_df['rank'] = range(1, len(successful_df) + 1)
        
        # Calculate cumulative weights (useful for MARK-style model selection)
        if f'{comparison_col}_weight' in successful_df.columns:
            successful_df['cumulative_weight'] = successful_df[f'{comparison_col}_weight'].cumsum()
        
        # Add MARK-style model selection indicators
        successful_df['substantial_support'] = successful_df[delta_col] <= 2.0  # Delta < 2
        successful_df['some_support'] = (successful_df[delta_col] > 2.0) & (successful_df[delta_col] <= 7.0)  # 2 < Delta <= 7
        successful_df['little_support'] = successful_df[delta_col] > 7.0  # Delta > 7
        
        return successful_df
    
    def print_summary(self, export_df: pd.DataFrame) -> None:
        """Print a comprehensive summary of exported results."""
        print(f"\n📊 Model Results Summary")
        print("=" * 50)
        
        successful_models = export_df[export_df['converged'] == True]
        failed_models = export_df[export_df['converged'] == False]
        
        print(f"Total models: {len(export_df)}")
        print(f"Successful: {len(successful_models)} ✅")
        print(f"Failed: {len(failed_models)} ❌")
        
        if len(successful_models) > 0:
            # Model comparison table
            comparison_df = self.create_model_comparison_table(export_df, sort_by='aic')
            
            print(f"\n🏆 Model Comparison (sorted by AIC):")
            display_cols = ['rank', 'model_name', 'aic', 'delta_aic', 'aic_weight', 'n_parameters']
            available_cols = [col for col in display_cols if col in comparison_df.columns]
            print(comparison_df[available_cols].to_string(index=False, float_format='%.4f'))
            
            # Parameter summary
            param_cols = [col for col in export_df.columns if col.startswith(('phi_', 'p_', 'f_'))]
            if param_cols:
                print(f"\n📋 Best Model Parameters:")
                best_model = comparison_df.iloc[0]
                print(f"   Model: {best_model['model_name']}")
                for col in param_cols:
                    if pd.notna(best_model.get(col)):
                        print(f"   {col}: {best_model[col]:.6f}")
            
            # Population dynamics
            if 'lambda_mean' in successful_models.columns:
                lambda_info = successful_models[['model_name', 'lambda_mean']].dropna()
                if len(lambda_info) > 0:
                    print(f"\n📈 Population Growth Rates (λ):")
                    print(lambda_info.to_string(index=False, float_format='%.4f'))
        
        if len(failed_models) > 0:
            print(f"\n❌ Failed Models:")
            failed_cols = ['model_name', 'error_message']
            available_failed_cols = [col for col in failed_cols if col in failed_models.columns]
            print(failed_models[available_failed_cols].to_string(index=False))

# Convenience functions for common use cases
def export_model_results(
    results: List[Union[ParallelOptimizationResult, ModelResult]],
    model_specs: Optional[List[ParallelModelSpec]] = None,
    data_context: Optional[DataContext] = None,
    export_file: Optional[Union[str, Path]] = None,
    print_summary: bool = True
) -> pd.DataFrame:
    """
    Convenience function to export model results with default settings.
    
    Args:
        results: List of optimization/model results
        model_specs: List of model specifications (if available)
        data_context: Data context used for fitting (if available) 
        export_file: Optional file path to save results
        print_summary: Whether to print a summary to console
        
    Returns:
        DataFrame with comprehensive model results
    """
    exporter = ResultsExporter()
    export_df = exporter.export_results(results, model_specs, data_context, export_file)
    
    if print_summary:
        exporter.print_summary(export_df)
    
    return export_df

def create_timestamped_export(
    results: List,
    model_specs: Optional[List] = None,
    data_context: Optional[DataContext] = None,
    prefix: str = "model_results",
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a timestamped export file for model results.
    
    Args:
        results: Model results to export
        model_specs: Model specifications (if available)
        data_context: Data context (if available)
        prefix: Filename prefix
        directory: Directory to save in (defaults to current directory)
        
    Returns:
        Path to created export file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    
    if directory:
        export_path = Path(directory) / filename
    else:
        export_path = Path(filename)
    
    export_model_results(results, model_specs, data_context, export_path)
    
    return export_path