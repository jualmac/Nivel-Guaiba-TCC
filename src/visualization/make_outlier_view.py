"""
Compare ECOD and PCA outlier detection algorithms by plotting decision boundaries.
Based on PyOD documentation example, adapted for the algorithms used in data_transformation.py.

This script visualizes how ECOD and PCA detect outliers on real data or synthetic data,
showing decision boundaries and classification results.

Available Functions:
    - compare_outlier_detectors(df, station_code, ...): Main function for real dataframes;
    - compare_outlier_detectors_synthetic(...): Synthetic 2D data comparison;
    - compare_multiple_contamination_levels(...): Sensitivity analysis across contamination levels;

Usage Example (in notebook):
    from source_frontend.make_outlier_view import compare_outlier_detectors
    
    # With real data;
    fig = compare_outlier_detectors(
        df=df_agg,
        station_code='87450004',
        threshold_method='iqr',
        max_display_points=5000,  # Limit visualization points for clarity;
        show_plot=True,
        save_plot=False
    )
"""

from __future__ import division, print_function
import os
import warnings
warnings.filterwarnings("ignore")

import logging
import numpy as np
import pandas as pd
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.decomposition import PCA as sklearn_PCA

# Import the models used in data_transformation.py;
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from util import configure_logging

logger = configure_logging(__name__)


def compare_outlier_detectors(df: pd.DataFrame = None, station_code: str = None,
                               threshold_method: str = 'iqr', max_display_points: int = 5000,
                               output_path='/home/juju/Documents/Nivel_TCC/data',
                               random_state=42, show_plot=True, save_plot=True):
    """
    Compare ECOD and PCA outlier detection algorithms on real dataframe.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with features to analyze;
        station_code (str): Optional station code to filter data (if None, uses all data);
        threshold_method (str): Method for threshold detection ('iqr' or 'percentile');
        max_display_points (int): Maximum number of points to display for clarity (default: 5000);
            Note: Statistics are computed on full dataset, only visualization is sampled;
        output_path (str): Directory path to save the output visualization;
        random_state (int): Random seed for reproducibility;
        show_plot (bool): Whether to display the plot (default: True);
        save_plot (bool): Whether to save the plot to file (default: True);
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure object;
    """
    # If no dataframe provided, use synthetic data;
    if df is None:
        return compare_outlier_detectors_synthetic(
            n_samples=200,
            outliers_fraction=0.25,
            output_path=output_path,
            random_state=random_state,
            show_plot=show_plot,
            save_plot=save_plot
        )
    
    # Copy dataframe to avoid modifications;
    df_cpy = df.copy()
    
    # Filter by station if specified;
    if station_code is not None:
        if 'codigoestacao' in df_cpy.columns:
            df_cpy = df_cpy[df_cpy['codigoestacao'] == station_code].copy()
            logger.info("Filtering data for station: %s", station_code)
        else:
            logger.warning("'codigoestacao' column not found, using all data")
    
    # Identify feature columns (exclude metadata and status columns);
    index_cols = ['Data_Hora_Medicao', 'codigoestacao']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols
    feature_cols = [col for col in df_cpy.columns if col not in non_feature_cols]
    
    logger.info("Using %s features: %s", len(feature_cols), feature_cols)
    
    # Extract features and remove rows with any missing values;
    df_features = df_cpy[feature_cols].copy()
    complete_mask = df_features.notna().all(axis=1)
    df_complete = df_features[complete_mask].copy()
    
    if len(df_complete) == 0:
        logger.error("No complete rows found in dataframe")
        return None
    
    logger.info("Using %s complete rows out of %s total rows", len(df_complete), len(df_cpy))
    logger.info("Missing data: %s rows (%.1f%%)", len(df_cpy) - len(df_complete), 100*(len(df_cpy)-len(df_complete))/len(df_cpy))
    
    # Convert to numpy array for PyOD;
    X = df_complete.values
    
    # Apply PCA for 2D visualization;
    pca_2d = sklearn_PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X)
    
    logger.info(
        "PCA projection: %.1f%% + %.1f%% = %.1f%% variance explained",
        pca_2d.explained_variance_ratio_[0]*100,
        pca_2d.explained_variance_ratio_[1]*100,
        sum(pca_2d.explained_variance_ratio_)*100
    )
    
    # Define the two detectors;
    detectors = {
        'ECOD (Empirical Cumulative Distribution)': ECOD(contamination=0.001),
        'PCA (Principal Component Analysis)': PCA(contamination=0.001, random_state=random_state)
    }
    
    # Create meshgrid for decision boundary visualization in 2D PCA space;
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))
    
    # Create figure with subplots;
    fig = plt.figure(figsize=(16, 7))
    
    # Store results from both detectors for comparison;
    results = {}
    
    for i, (clf_name, clf) in enumerate(detectors.items()):
        logger.info("[%s/%s] Fitting %s...", i+1, len(detectors), clf_name)
        
        # Fit detector on full-dimensional data;
        clf.fit(X)
        
        # Get outlier scores and predictions;
        scores = clf.decision_scores_
        
        # Apply dynamic threshold detection;
        if threshold_method == 'iqr':
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        elif threshold_method == 'percentile':
            threshold = np.percentile(scores, 95)
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
        
        predictions = (scores > threshold).astype(int)
        n_outliers = predictions.sum()
        outlier_rate = n_outliers / len(predictions) * 100
        
        logger.info("Threshold: %.4f", threshold)
        logger.info("Detected outliers: %s/%s (%.2f%%)", n_outliers, len(predictions), outlier_rate)
        
        # Store results;
        results[clf_name] = {
            'predictions': predictions,
            'threshold': threshold,
            'n_outliers': n_outliers,
            'outlier_rate': outlier_rate
        }
        
        # For visualization, we need to project the meshgrid back to original space;
        # This is an approximation - we inverse transform 2D PCA back to original features;
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_points_full = pca_2d.inverse_transform(grid_points_2d)
        
        # Get decision scores for grid;
        Z = clf.decision_function(grid_points_full)
        Z = Z.reshape(xx.shape)
        
        # Create subplot;
        subplot = plt.subplot(1, 2, i + 1)
        
        # Plot decision boundary with simpler color scheme;
        levels = np.linspace(Z.min(), threshold, 5)
        subplot.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Blues_r, alpha=0.6)
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange', alpha=0.4)
        
        # Sample BOTH outliers and normal points intelligently for visualization;
        normal_mask = predictions == 0
        outlier_mask = predictions == 1
        
        outlier_indices = np.where(outlier_mask)[0]
        normal_indices = np.where(normal_mask)[0]
        
        # Cap maximum outliers to display to reduce clutter;
        max_outliers_display = min(len(outlier_indices), 2000)  # Hard cap at 2000 outliers;
        max_normal_display = max(1000, max_display_points - max_outliers_display)
        
        # Sample both populations;
        np.random.seed(random_state)
        if len(outlier_indices) > max_outliers_display:
            outlier_sample_indices = np.random.choice(outlier_indices, max_outliers_display, replace=False)
            logger.info("Displaying: %s sampled outliers (of %s total)", max_outliers_display, len(outlier_indices))
        else:
            outlier_sample_indices = outlier_indices
            logger.info("Displaying: %s outliers", len(outlier_indices))
        
        if len(normal_indices) > max_normal_display:
            normal_sample_indices = np.random.choice(normal_indices, max_normal_display, replace=False)
            logger.info("%s sampled normal points (of %s total)", max_normal_display, len(normal_indices))
        else:
            normal_sample_indices = normal_indices
            logger.info("%s normal points", len(normal_indices))
        
        # Plot sampled normal points with reduced visual weight;
        subplot.scatter(
            X_2d[normal_sample_indices, 0], X_2d[normal_sample_indices, 1],
            c='blue', s=5, alpha=0.3, edgecolor='none',
            label=f'Normal ({normal_mask.sum():,})', zorder=5
        )
        
        # Plot sampled outliers with reduced clutter;
        if len(outlier_sample_indices) > 0:
            subplot.scatter(
                X_2d[outlier_sample_indices, 0], X_2d[outlier_sample_indices, 1],
                c='red', s=20, alpha=0.7, edgecolor='darkred', linewidths=0.5,
                label=f'Outliers ({len(outlier_indices):,})', zorder=6, marker='x'
            )
        
        # Configure subplot;
        subplot.set_xlim((x_min, x_max))
        subplot.set_ylim((y_min, y_max))
        subplot.set_xlabel(
            f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)",
            fontsize=11
        )
        subplot.set_ylabel(
            f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)",
            fontsize=11
        )
        subplot.set_title(
            f"{clf_name.split('(')[0].strip()}\n"
            f"Outliers: {n_outliers:,} ({outlier_rate:.2f}%)",
            fontsize=12, fontweight='bold'
        )
        subplot.legend(loc='best', fontsize=10, framealpha=0.9)
        subplot.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Add main title;
    station_text = f" - Station {station_code}" if station_code else ""
    plt.suptitle(
        f'Outlier Detection Comparison: ECOD vs PCA{station_text}\n'
        f'Dataset: {len(df_complete):,} samples, {len(feature_cols)} features | '
        f'Threshold Method: {threshold_method.upper()}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if requested;
    if save_plot:
        os.makedirs(output_path, exist_ok=True)
        filename = f'outlier_detection_comparison_{"station_" + station_code if station_code else "all"}.png'
        output_file = os.path.join(output_path, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Visualization saved: %s", output_file)
    
    # Display plot if requested;
    if show_plot:
        plt.show()
    return fig


def compare_outlier_detectors_synthetic(n_samples=200, outliers_fraction=0.25, 
                                         output_path='/home/juju/Documents/Nivel_TCC/data',
                                         random_state=42, show_plot=True, save_plot=True):
    """
    Compare ECOD and PCA outlier detection algorithms on synthetic 2D data.
    
    Parameters:
        n_samples (int): Total number of samples to generate;
        outliers_fraction (float): Fraction of outliers in the dataset;
        output_path (str): Directory path to save the output visualization;
        random_state (int): Random seed for reproducibility;
        show_plot (bool): Whether to display the plot (default: True);
        save_plot (bool): Whether to save the plot to file (default: True);
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure object;
    """
    logger.info("=" * 80)
    logger.info("OUTLIER DETECTION ALGORITHM COMPARISON (SYNTHETIC DATA)")
    logger.info("=" * 80)
    
    # Create meshgrid for decision boundary visualization;
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
    
    # Calculate number of inliers and outliers;
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    
    # Create ground truth labels (0 = inlier, 1 = outlier);
    ground_truth = np.zeros(n_samples, dtype=int)
    ground_truth[-n_outliers:] = 1
    
    # Display dataset statistics;
    logger.info("Number of inliers: %s", n_inliers)
    logger.info("Number of outliers: %s", n_outliers)
    logger.info("Outliers fraction: %.2f%%", outliers_fraction * 100)
    logger.info("Ground truth shape: %s", ground_truth.shape)
    
    # Define the two detectors used in data_transformation.py;
    classifiers = {
        'ECOD (Empirical Cumulative Distribution)': ECOD(
            contamination=outliers_fraction
        ),
        'PCA (Principal Component Analysis)': PCA(
            contamination=outliers_fraction, 
            random_state=random_state
        )
    }
    
    # Generate synthetic 2D data;
    np.random.seed(random_state)
    
    # Create two inlier clusters with Gaussian distribution;
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2)
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2)
    X_inliers = np.r_[X1, X2]
    
    # Add uniformly distributed outliers in the feature space;
    X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))
    X = np.r_[X_inliers, X_outliers]
    
    logger.info("Data generation complete. Fitting models...")
    
    # Create figure with subplots for each detector;
    fig = plt.figure(figsize=(16, 7))
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        logger.info("[%s/%s] Fitting %s...", i+1, len(classifiers), clf_name)
        
        # Fit the detector on the data;
        clf.fit(X)
        
        # Get outlier scores (invert sign for visualization);
        scores_pred = clf.decision_function(X) * -1
        
        # Get binary predictions (0 = inlier, 1 = outlier);
        y_pred = clf.predict(X)
        
        # Calculate threshold at the specified outliers_fraction percentile;
        threshold = percentile(scores_pred, 100 * outliers_fraction)
        
        # Calculate number of classification errors;
        n_errors = (y_pred != ground_truth).sum()
        error_rate = n_errors / n_samples * 100
        
        logger.info("Threshold: %.3f", threshold)
        logger.info("Errors: %s/%s (%.1f%%)", n_errors, n_samples, error_rate)
        logger.info("Detected outliers: %s/%s", y_pred.sum(), n_samples)
        
        # Compute decision function on meshgrid for decision boundary visualization;
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)
        
        # Create subplot for this detector;
        subplot = plt.subplot(1, 2, i + 1)
        
        # Plot decision boundary using contour fill;
        # Blue gradient for normal region;
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                        cmap=plt.cm.Blues_r, alpha=0.8)
        
        # Orange region for outlier zone;
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                        colors='orange', alpha=0.6)
        
        # Plot true inliers as white points;
        inliers_scatter = subplot.scatter(
            X[:-n_outliers, 0], X[:-n_outliers, 1], 
            c='white', s=30, edgecolor='k', linewidths=1,
            label='True Inliers', zorder=5
        )
        
        # Plot true outliers as black points;
        outliers_scatter = subplot.scatter(
            X[-n_outliers:, 0], X[-n_outliers:, 1], 
            c='black', s=30, edgecolor='k', linewidths=1,
            label='True Outliers', zorder=5
        )
        
        # Configure subplot appearance;
        subplot.axis('tight')
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
        subplot.set_xlabel(
            f"{clf_name}\nErrors: {n_errors}/{n_samples} ({error_rate:.1f}%)", 
            fontsize=11, fontweight='bold'
        )
        subplot.legend(
            [inliers_scatter, outliers_scatter],
            ['True Inliers', 'True Outliers'],
            loc='upper right',
            fontsize=10
        )
        subplot.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add main title;
    plt.suptitle(
        'Outlier Detection Comparison: ECOD vs PCA\n'
        f'Dataset: {n_samples} samples ({outliers_fraction:.0%} outliers)',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if requested;
    if save_plot:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'outlier_detection_comparison_synthetic.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Visualization saved: %s", output_file)
    
    # Display plot if requested;
    if show_plot:
        plt.show()
    
    logger.info("=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    
    return fig


def compare_multiple_contamination_levels(output_path='/home/juju/Documents/Nivel_TCC/data',
                                           random_state=42, show_plot=True, save_plot=True):
    """
    Extended comparison showing how detectors perform at different contamination levels.
    Useful for understanding algorithm behavior across various outlier fractions.
    
    Parameters:
        output_path (str): Directory path to save the output visualization;
        random_state (int): Random seed for reproducibility;
        show_plot (bool): Whether to display the plot (default: True);
        save_plot (bool): Whether to save the plot to file (default: True);
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure object;
    """
    logger.info("=" * 80)
    logger.info("MULTI-CONTAMINATION ANALYSIS")
    logger.info("=" * 80)
    
    contamination_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
    n_samples = 200
    random_state = 42
    
    # Create figure for multi-contamination comparison;
    fig, axes = plt.subplots(len(contamination_levels), 2, figsize=(16, 5 * len(contamination_levels)))
    
    for row_idx, contamination in enumerate(contamination_levels):
        logger.info("Processing contamination level: %.0f%%", contamination * 100)
        
        # Calculate sample distribution;
        n_inliers = int((1. - contamination) * n_samples)
        n_outliers = int(contamination * n_samples)
        ground_truth = np.zeros(n_samples, dtype=int)
        ground_truth[-n_outliers:] = 1
        
        # Generate data;
        np.random.seed(random_state)
        X1 = 0.3 * np.random.randn(n_inliers // 2, 2)
        X2 = 0.3 * np.random.randn(n_inliers // 2, 2)
        X_inliers = np.r_[X1, X2]
        X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))
        X = np.r_[X_inliers, X_outliers]
        
        # Meshgrid for decision boundaries;
        xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
        
        # Test both detectors;
        classifiers = [
            ('ECOD', ECOD(contamination=contamination)),
            ('PCA', PCA(contamination=contamination, random_state=random_state))
        ]
        
        for col_idx, (clf_name, clf) in enumerate(classifiers):
            # Fit and predict;
            clf.fit(X)
            scores_pred = clf.decision_function(X) * -1
            y_pred = clf.predict(X)
            threshold = percentile(scores_pred, 100 * contamination)
            n_errors = (y_pred != ground_truth).sum()
            
            # Compute decision boundary;
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
            Z = Z.reshape(xx.shape)
            
            # Plot in grid;
            ax = axes[row_idx, col_idx]
            ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                       cmap=plt.cm.Blues_r, alpha=0.8)
            ax.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                       colors='orange', alpha=0.6)
            ax.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], 
                      c='white', s=20, edgecolor='k', linewidths=0.8)
            ax.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], 
                      c='black', s=20, edgecolor='k', linewidths=0.8)
            ax.set_xlim((-7, 7))
            ax.set_ylim((-7, 7))
            ax.set_title(
                f"{clf_name} | Contamination: {contamination:.0%} | Errors: {n_errors}",
                fontsize=10, fontweight='bold'
            )
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.suptitle(
        'Outlier Detection Sensitivity to Contamination Level',
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure if requested;
    if save_plot:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'outlier_detection_multi_contamination.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info("Multi-contamination analysis saved: %s", output_file)
    
    # Display plot if requested;
    if show_plot:
        plt.show()
    
    logger.info("=" * 80)
    logger.info("MULTI-CONTAMINATION ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return fig


########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    # Example: Load data and run comparison;
    # Uncomment and modify the following to use with real data:
    
    # from src.etl.data_transformation import collect_all_stations
    # df_cleaned, df_filled, df_agg, df_out, df_imp, df_melted, imputer_stats = collect_all_stations(
    #     save_to_db=False, 
    #     frequency='h', 
    #     max_fill_steps=8
    # )
    
    # # Compare outlier detectors on real data for a specific station;
    # fig1 = compare_outlier_detectors(
    #     df=df_agg,
    #     station_code='87450004',  # Guaiba station;
    #     threshold_method='iqr',
    #     output_path='/home/juju/Documents/Nivel_TCC/data',
    #     random_state=42,
    #     show_plot=True,
    #     save_plot=True
    # )
    
    # Or run with synthetic data (no df provided);
    fig1 = compare_outlier_detectors_synthetic(
        n_samples=200,
        outliers_fraction=0.25,
        output_path='/home/juju/Documents/Nivel_TCC/data',
        random_state=42,
        show_plot=True,
        save_plot=True
    )
    
    # Extended analysis across multiple contamination levels (synthetic data);
    fig2 = compare_multiple_contamination_levels(
        output_path='/home/juju/Documents/Nivel_TCC/data',
        random_state=42,
        show_plot=True,
        save_plot=True
    )