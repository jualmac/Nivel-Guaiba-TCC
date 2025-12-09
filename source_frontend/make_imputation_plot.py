"""
Imputation Visualization Module

Provides diagnostic plots to analyze the quality and behavior of feature interpolation
performed by the CubicSpline-based imputer. Helps identify fill patterns, temporal
behavior, and station-specific interpolation performance.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from util import configure_logging

logger = configure_logging(__name__)


########################################################################################################################
#                                                                  
# FUNCTIONS
#
########################################################################################################################
def plot_imputation_summary(imputer_stats: Dict, figsize: Tuple[int, int] = (16, 10)):
    """
    Create overview dashboard showing imputation statistics across all stations;
    
    Parameters:
        imputer_stats (dict): Dictionary with station-level imputation statistics;
        figsize (tuple): Figure size (width, height);
    
    Returns:
        fig, axes: Matplotlib figure and axes objects;
    """
    # Extract stats;
    stations = []
    n_features_imputed = []
    n_features_skipped = []
    n_iterations = []
    
    for station, stats in imputer_stats.items():
        if 'error' not in stats:
            stations.append(station)
            n_features_imputed.append(len(stats['cols_imputed']))
            n_features_skipped.append(len(stats['cols_skipped']))
            n_iterations.append(stats['n_iter'])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Imputation Summary Across Stations', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of features imputed vs skipped per station;
    x_pos = np.arange(len(stations))
    axes[0, 0].bar(x_pos - 0.2, n_features_imputed, 0.4, label='Imputed', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x_pos + 0.2, n_features_skipped, 0.4, label='Skipped (100% missing)', alpha=0.8, color='coral')
    axes[0, 0].set_xlabel('Station ID')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Features Imputed vs Skipped per Station')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(stations, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Convergence iterations per station;
    axes[0, 1].bar(stations, n_iterations, alpha=0.8, color='forestgreen')
    axes[0, 1].set_xlabel('Station ID')
    axes[0, 1].set_ylabel('Iterations to Convergence')
    axes[0, 1].set_title('Imputation Convergence Speed')
    axes[0, 1].set_xticklabels(stations, rotation=45, ha='right')
    axes[0, 1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Max iterations')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Initial mean values distribution;
    axes[1, 0].axis('off')
    text_content = "Imputation Statistics:\n\n"
    for i, station in enumerate(stations):
        stats = imputer_stats[station]
        text_content += f"Station {station}:\n"
        text_content += f"  - Features imputed: {len(stats['cols_imputed'])}\n"
        text_content += f"  - Iterations: {stats['n_iter']}\n"
        if i < len(stations) - 1:
            text_content += "\n"
    axes[1, 0].text(0.1, 0.9, text_content, transform=axes[1, 0].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Plot 4: Imputation sequence order (feature ordering);
    axes[1, 1].axis('off')
    text_content = "Imputation Order (first 3 rounds):\n\n"
    for station in stations[:3]:  # Show first 3 stations;
        stats = imputer_stats[station]
        if 'imputation_sequence' in stats:
            seq = stats['imputation_sequence'][:5]  # First 5 features;
            feature_names = [stats['cols_imputed'][idx] if idx < len(stats['cols_imputed']) else 'N/A' for idx in seq]
            text_content += f"{station}: {', '.join(feature_names)}\n"
    axes[1, 1].text(0.1, 0.9, text_content, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig, axes


def plot_initial_means_comparison(imputer_stats: Dict, figsize: Tuple[int, int] = (14, 8)):
    """
    Compare initial mean values used for imputation across stations and features;
    
    Parameters:
        imputer_stats (dict): Dictionary with station-level imputation statistics;
        figsize (tuple): Figure size (width, height);
    
    Returns:
        fig, ax: Matplotlib figure and axis objects;
    """
    # Collect initial means per feature across stations;
    feature_means = {}
    
    for station, stats in imputer_stats.items():
        if 'initial_means' in stats:
            for feature, mean_val in stats['initial_means'].items():
                if feature not in feature_means:
                    feature_means[feature] = {}
                feature_means[feature][station] = mean_val
    
    # Convert to DataFrame for easier plotting;
    df_means = pd.DataFrame(feature_means).T
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_means, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Initial Mean Value'})
    ax.set_title('Initial Mean Values Used for Imputation', fontsize=14, fontweight='bold')
    ax.set_xlabel('Station ID')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    return fig, ax


def plot_imputed_vs_original_timeseries(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame, 
    station_id: str, 
    feature: str = 'Cota_Adotada',
    date_col: str = 'Data_Hora_Medicao',
    figsize: Tuple[int, int] = (16, 6)
):
    """
    Plot time series comparison showing original data with gaps vs imputed data;
    
    Parameters:
        df_before (pd.DataFrame): Dataframe before imputation;
        df_after (pd.DataFrame): Dataframe after imputation;
        station_id (str): Station code to filter;
        feature (str): Feature column to visualize;
        date_col (str): Date column name;
        figsize (tuple): Figure size (width, height);
    
    Returns:
        fig, ax: Matplotlib figure and axis objects;
    """
    # Filter data for specific station;
    df_before_station = df_before[df_before['codigoestacao'] == station_id].copy()
    df_after_station = df_after[df_after['codigoestacao'] == station_id].copy()
    
    # Sort by date;
    df_before_station = df_before_station.sort_values(date_col)
    df_after_station = df_after_station.sort_values(date_col)
    
    # Identify imputed points (was NaN before, has value after);
    was_missing = df_before_station[feature].isna()
    imputed_mask = was_missing & df_after_station[feature].notna()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data;
    ax.plot(df_before_station[date_col], df_before_station[feature], 
            label='Original Data', color='steelblue', linewidth=1.5, alpha=0.7)
    
    # Highlight imputed points;
    imputed_dates = df_after_station.loc[imputed_mask, date_col]
    imputed_values = df_after_station.loc[imputed_mask, feature]
    ax.scatter(imputed_dates, imputed_values, 
               label='Imputed Values', color='red', s=10, alpha=0.6, zorder=5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} Time Series for Station {station_id}\nOriginal vs Imputed', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_imputation_gap_distribution(
    df_before: pd.DataFrame, 
    df_after: pd.DataFrame,
    feature: str = 'Cota_Adotada',
    date_col: str = 'Data_Hora_Medicao',
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Analyze and visualize the distribution of gap sizes that were imputed;
    
    Parameters:
        df_before (pd.DataFrame): Dataframe before imputation;
        df_after (pd.DataFrame): Dataframe after imputation;
        feature (str): Feature column to analyze;
        date_col (str): Date column name;
        figsize (tuple): Figure size (width, height);
    
    Returns:
        fig, axes: Matplotlib figure and axes objects;
    """
    gap_info = []
    
    for station_id in df_before['codigoestacao'].unique():
        df_before_station = df_before[df_before['codigoestacao'] == station_id].copy()
        df_after_station = df_after[df_after['codigoestacao'] == station_id].copy()
        
        df_before_station = df_before_station.sort_values(date_col).reset_index(drop=True)
        df_after_station = df_after_station.sort_values(date_col).reset_index(drop=True)
        
        # Find imputed regions;
        was_missing = df_before_station[feature].isna()
        is_imputed = was_missing & df_after_station[feature].notna()
        
        # Calculate gap sizes;
        gap_lengths = []
        current_gap = 0
        
        for i, missing in enumerate(was_missing):
            if missing:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_lengths.append(current_gap)
                    current_gap = 0
        
        if current_gap > 0:
            gap_lengths.append(current_gap)
        
        for gap_len in gap_lengths:
            gap_info.append({
                'station_id': station_id,
                'gap_length': gap_len
            })
    
    df_gaps = pd.DataFrame(gap_info)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Gap Distribution Analysis for {feature}', fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram of gap lengths;
    axes[0].hist(df_gaps['gap_length'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Gap Length (number of consecutive missing points)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Gap Sizes')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot per station;
    station_groups = df_gaps.groupby('station_id')['gap_length'].apply(list)
    axes[1].boxplot(station_groups.values, labels=station_groups.index)
    axes[1].set_xlabel('Station ID')
    axes[1].set_ylabel('Gap Length')
    axes[1].set_title('Gap Size Distribution per Station')
    axes[1].set_xticklabels(station_groups.index, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_imputed_values_distribution(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature: str = 'Cota_Adotada',
    station_id: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Compare distribution of original values vs imputed values to detect mean-fill bias;
    
    Parameters:
        df_before (pd.DataFrame): Dataframe before imputation;
        df_after (pd.DataFrame): Dataframe after imputation;
        feature (str): Feature column to analyze;
        station_id (str, optional): Specific station to analyze, or None for all;
        figsize (tuple): Figure size (width, height);
    
    Returns:
        fig, axes: Matplotlib figure and axes objects;
    """
    if station_id:
        df_before = df_before[df_before['codigoestacao'] == station_id].copy()
        df_after = df_after[df_after['codigoestacao'] == station_id].copy()
        title_suffix = f' - Station {station_id}'
    else:
        title_suffix = ' - All Stations'
    
    # Identify original vs imputed values;
    was_missing = df_before[feature].isna()
    original_values = df_before.loc[~was_missing, feature].dropna()
    imputed_values = df_after.loc[was_missing, feature].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{feature} Distribution: Original vs Imputed{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Overlapping histograms;
    axes[0].hist(original_values, bins=50, alpha=0.6, label='Original', color='steelblue', density=True)
    axes[0].hist(imputed_values, bins=50, alpha=0.6, label='Imputed', color='coral', density=True)
    axes[0].axvline(original_values.mean(), color='steelblue', linestyle='--', linewidth=2, 
                    label=f'Original Mean: {original_values.mean():.2f}')
    axes[0].axvline(imputed_values.mean(), color='coral', linestyle='--', linewidth=2,
                    label=f'Imputed Mean: {imputed_values.mean():.2f}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution Comparison')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Box plots;
    data_to_plot = [original_values, imputed_values]
    axes[1].boxplot(data_to_plot, labels=['Original', 'Imputed'])
    axes[1].set_ylabel(feature)
    axes[1].set_title('Box Plot Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistical info;
    stats_text = f"Original: μ={original_values.mean():.2f}, σ={original_values.std():.2f}\n"
    stats_text += f"Imputed: μ={imputed_values.mean():.2f}, σ={imputed_values.std():.2f}\n"
    stats_text += f"Mean difference: {abs(original_values.mean() - imputed_values.mean()):.2f}"
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, axes


def create_imputation_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    imputer_stats: Dict,
    output_path: Optional[str] = None,
    features: Optional[List[str]] = None
):
    """
    Generate comprehensive imputation report with multiple diagnostic plots;
    
    Parameters:
        df_before (pd.DataFrame): Dataframe before imputation;
        df_after (pd.DataFrame): Dataframe after imputation;
        imputer_stats (dict): Imputer statistics from feature_imputation function;
        output_path (str, optional): Path to save report figures, or None to just display;
        features (list, optional): List of features to analyze, or None for defaults;
    
    Returns:
        dict: Dictionary containing all generated figures;
    """
    if features is None:
        features = ['Cota_Adotada', 'Vazao_Adotada', 'Temperatura_Interna']
    
    figures = {}
    
    # 1. Summary dashboard;
    logger.info("Generating imputation summary...")
    fig_summary, _ = plot_imputation_summary(imputer_stats)
    figures['summary'] = fig_summary
    if output_path:
        fig_summary.savefig(f"{output_path}/imputation_summary.png", dpi=150, bbox_inches='tight')
    
    # 2. Initial means comparison;
    logger.info("Generating initial means comparison...")
    fig_means, _ = plot_initial_means_comparison(imputer_stats)
    figures['means'] = fig_means
    if output_path:
        fig_means.savefig(f"{output_path}/initial_means.png", dpi=150, bbox_inches='tight')
    
    # 3. Time series comparisons per feature;
    for feature in features:
        if feature in df_before.columns:
            logger.info("Generating time series for %s...", feature)
            for station_id in df_before['codigoestacao'].unique()[:3]:  # First 3 stations;
                try:
                    fig_ts, _ = plot_imputed_vs_original_timeseries(
                        df_before, df_after, station_id, feature
                    )
                    figures[f'timeseries_{feature}_{station_id}'] = fig_ts
                    if output_path:
                        fig_ts.savefig(
                            f"{output_path}/timeseries_{feature}_{station_id}.png", 
                            dpi=150, bbox_inches='tight'
                        )
                except Exception as e:
                    logger.warning("Could not plot %s for station %s: %s", feature, station_id, e)
    
    # 4. Gap distribution analysis;
    for feature in features:
        if feature in df_before.columns:
            logger.info("Generating gap distribution for %s...", feature)
            try:
                fig_gaps, _ = plot_imputation_gap_distribution(df_before, df_after, feature)
                figures[f'gaps_{feature}'] = fig_gaps
                if output_path:
                    fig_gaps.savefig(f"{output_path}/gaps_{feature}.png", dpi=150, bbox_inches='tight')
            except Exception as e:
                logger.warning("Could not analyze gaps for %s: %s", feature, e)
    
    # 5. Distribution comparison;
    for feature in features:
        if feature in df_before.columns:
            logger.info("Generating distribution comparison for %s...", feature)
            for station_id in df_before['codigoestacao'].unique()[:3]:  # First 3 stations;
                try:
                    fig_dist, _ = plot_imputed_values_distribution(
                        df_before, df_after, feature, station_id
                    )
                    figures[f'distribution_{feature}_{station_id}'] = fig_dist
                    if output_path:
                        fig_dist.savefig(
                            f"{output_path}/distribution_{feature}_{station_id}.png", 
                            dpi=150, bbox_inches='tight'
                        )
                except Exception as e:
                    logger.warning("Could not plot distribution for %s at %s: %s", feature, station_id, e)
    
    logger.info("Generated %s figures for imputation analysis.", len(figures))
    return figures


########################################################################################################################
#
# EXAMPLE USAGE
#
########################################################################################################################
if __name__ == "__main__":
    logger.info("Imputation visualization module loaded.")
    logger.info("Use the following functions:")
    logger.info("  - plot_imputation_summary(imputer_stats)")
    logger.info("  - plot_initial_means_comparison(imputer_stats)")
    logger.info("  - plot_imputed_vs_original_timeseries(df_before, df_after, station_id, feature)")
    logger.info("  - plot_imputation_gap_distribution(df_before, df_after, feature)")
    logger.info("  - plot_imputed_values_distribution(df_before, df_after, feature, station_id)")
    logger.info("  - create_imputation_report(df_before, df_after, imputer_stats, output_path)")

