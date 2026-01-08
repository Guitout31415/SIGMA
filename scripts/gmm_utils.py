"""
gmm_utils.py
------------
Gaussian Mixture Model utilities for the SIGMA pipeline.

Provides functions for GMM fitting, KDE-based component estimation,
cross-validation for bandwidth selection, and target component identification.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from constants import (
    KDE_GRID_POINTS,
    PEAK_PROMINENCE,
    SECOND_DERIVATIVE_THRESHOLD,
    ASHMANN_DISTANCE_THRESHOLD,
    CV_FOLDS,
    CV_TEST_SIZE,
)


# =============================================================================
# KDE Bandwidth Selection
# =============================================================================


def kde_cross_validation(
    data: np.ndarray,
    bw_candidates: np.ndarray,
    cv_folds: int = CV_FOLDS,
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Find optimal KDE bandwidth using cross-validation.

    Uses log-likelihood scoring on held-out data to select the bandwidth
    that best generalizes to unseen data.

    Args:
        data: 1D array of data points
        bw_candidates: Array of bandwidth values to test
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (optimal_bandwidth, cv_scores_dict)
            where cv_scores_dict maps bandwidth to {'mean': ..., 'std': ...}
    """
    cv_scores = {}

    for bw in bw_candidates:
        fold_scores = []

        for _ in range(cv_folds):
            train_data, test_data = train_test_split(
                data, test_size=CV_TEST_SIZE, random_state=None
            )

            try:
                kde_train = gaussian_kde(train_data, bw_method=bw)
                log_likelihood = np.sum(np.log(kde_train(test_data) + 1e-10))
                fold_scores.append(log_likelihood)
            except Exception:
                fold_scores.append(-np.inf)

        cv_scores[bw] = {"mean": np.mean(fold_scores), "std": np.std(fold_scores)}

    optimal_bw = max(
        cv_scores.keys(),
        key=lambda x: cv_scores[x]["mean"] - cv_scores[x]["std"]
    )
    return optimal_bw, cv_scores


# =============================================================================
# Component Estimation
# =============================================================================


def find_optimal_gmm_components(
    data: np.ndarray,
    exclude_celltypes: str,
    category: str,
) -> Tuple[int, Optional[np.ndarray]]:
    """Determine optimal number of GMM components using KDE peak detection.

    Uses kernel density estimation to find peaks in the expression
    distribution, then filters peaks by second derivative significance.

    Args:
        data: Expression data array
        exclude_celltypes: Whether to exclude entire cell types ('True'/'False')
        category: Category name ('Target' or exclusion category)

    Returns:
        Tuple of (n_components, initial_means)
            n_components: Estimated number of GMM components
            initial_means: Optional initial mean estimates for GMM initialization
    """
    data = data.flatten()
    data = data[np.isfinite(data) & (data > 0.2)]

    if len(data) < 2 or np.var(data) == 0:
        return 1, None

    # Find optimal bandwidth
    bw_candidates = np.logspace(-1, 0.3, 150)
    optimal_bw, _ = kde_cross_validation(data, bw_candidates, cv_folds=CV_FOLDS)

    # Compute KDE and find peaks
    kde_optimal = gaussian_kde(data, bw_method=optimal_bw)
    x_grid = np.linspace(data.min(), data.max(), KDE_GRID_POINTS)
    y_grid = kde_optimal(x_grid)

    peaks, _ = find_peaks(y_grid, prominence=PEAK_PROMINENCE)

    if len(peaks) > 0:
        # Filter peaks by second derivative
        spline_interp = UnivariateSpline(x_grid, y_grid, s=0, k=3)
        deriv2 = spline_interp.derivative().derivative()

        significant_peaks = [
            idx for idx in peaks if deriv2(x_grid[idx]) < SECOND_DERIVATIVE_THRESHOLD
        ]
        peaks = np.array(significant_peaks)

        n_peaks = len(peaks) if len(peaks) > 0 else 1
        estimated_means = x_grid[peaks].reshape(-1, 1) if len(peaks) > 0 else None
    else:
        n_peaks = 1
        estimated_means = None

    # Add component for exclusion genes if not excluding whole cell types
    if exclude_celltypes == "False" and category != "Target":
        n_peaks += 1
        if estimated_means is not None:
            estimated_means = np.vstack([estimated_means, np.array([0])])

    return n_peaks, estimated_means


# =============================================================================
# GMM Fitting
# =============================================================================


def fit_gmm_and_predict_probas(
    data: np.ndarray,
    n_components: str,
    category: str,
    exclude_celltypes: str,
) -> Tuple[GaussianMixture, np.ndarray]:
    """Fit GMM and compute posterior probabilities.

    Args:
        data: Expression data array
        n_components: Number of components ('auto' or integer string)
        category: Category name for logging
        exclude_celltypes: Whether to exclude entire cell types ('True'/'False')

    Returns:
        Tuple of (fitted_gmm, probabilities)
            fitted_gmm: Fitted GaussianMixture model
            probabilities: Posterior probability matrix (n_cells x n_components)
    """
    if n_components == "auto":
        print(f"Automatically determining components for {category}...")
        optimal_n, estimated_means = find_optimal_gmm_components(
            data, exclude_celltypes, category
        )
        print(f"\tOptimal number of components: {optimal_n}")
    else:
        optimal_n, estimated_means = int(n_components), None
        print(f"Using specified components for {category}: {optimal_n}")

    # Filter data for fitting only (training data)
    data_flat = data.flatten()
    mask = np.isfinite(data_flat) & (data_flat > 0.2)
    data_train = data_flat[mask].reshape(-1, 1)

    gmm = GaussianMixture(n_components=optimal_n, means_init=estimated_means)
    gmm.fit(data_train)
    
    # Predict probabilities for ALL cells (not just filtered)
    data_all = data_flat.reshape(-1, 1)
    probas = gmm.predict_proba(data_all)

    return gmm, probas


# =============================================================================
# Distance Metrics
# =============================================================================


def ashmann_distance(m1: float, m2: float, s1: float, s2: float) -> float:
    """Compute Ashmann distance between two Gaussian distributions.

    The Ashmann distance measures separation between two Gaussians,
    useful for determining if GMM components represent distinct populations.

    Args:
        m1: Mean of first distribution
        m2: Mean of second distribution
        s1: Standard deviation of first distribution
        s2: Standard deviation of second distribution

    Returns:
        Ashmann distance value (>2 typically indicates good separation)
    """
    return np.abs(m1 - m2) / np.sqrt(s1**2 + s2**2)


# =============================================================================
# Target Identification
# =============================================================================


def identify_target_components(
    gmm: GaussianMixture,
    probas: np.ndarray,
    min_mean_expression: float,
) -> np.ndarray:
    """Identify components belonging to the target population.

    Uses the component with highest mean as the primary target,
    then merges nearby components based on Ashmann distance.

    Args:
        gmm: Fitted Gaussian Mixture Model
        probas: Predicted probabilities (n_cells x n_components)
        min_mean_expression: Minimum expression threshold for target

    Returns:
        Array of target probabilities per cell (sum of target component probs)
    """
    target_component = np.argmax(gmm.means_.flatten())

    if gmm.means_.flatten()[target_component] < min_mean_expression:
        print(
            f"Target component mean ({gmm.means_.flatten()[target_component]:.4f}) "
            f"is below threshold ({min_mean_expression})."
        )
        return np.zeros(probas.shape[0])

    # Find components close to target using Ashmann distance
    means_components = [float(m) for m in gmm.means_.flatten()]
    target_indices = [int(target_component)]
    means_components[target_component] = -1

    target_before = np.argmax(means_components)
    m_t = gmm.means_[target_component][0]
    s_t = np.sqrt(gmm.covariances_[target_component][0][0])

    while target_before > 0 and means_components[target_before] >= min_mean_expression:
        m_b = gmm.means_[target_before][0]
        s_b = np.sqrt(gmm.covariances_[target_before][0][0])

        if ashmann_distance(m_b, m_t, s_b, s_t) > ASHMANN_DISTANCE_THRESHOLD:
            break

        target_indices.append(target_before)
        means_components[target_before] = -1
        target_before = np.argmax(means_components)

    return np.sum(probas[:, target_indices], axis=1)
