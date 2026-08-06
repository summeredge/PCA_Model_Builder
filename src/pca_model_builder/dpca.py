from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f, norm
from sklearn.decomposition import PCA

from .scoring_core import score_dynamic_feature_matrix


@dataclass(frozen=True)
class DPCAModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    t2_limits: dict[float, float]
    q_limits: dict[float, float]
    n_samples: int

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def standardize(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        values = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("model inputs contain non-finite values")
        return (values - self.mean) / self.scale

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        scored = score_dynamic_feature_matrix(
            frame.loc[:, self.feature_names].to_numpy(dtype=float),
            feature_names=self.feature_names,
            mean=self.mean,
            scale=self.scale,
            components=self.components,
            eigenvalues=self.eigenvalues,
            t2_limits=self.t2_limits,
            q_limits=self.q_limits,
        )
        result = pd.DataFrame(index=frame.index)
        for index in range(self.n_components):
            result[f"pc{index + 1}"] = scored.pc_scores[:, index]
        result["t2"] = scored.t2
        result["spe"] = scored.spe
        result["t2_limit_ratio"] = scored.t2_limit_ratio
        result["spe_limit_ratio"] = scored.spe_limit_ratio
        result["t2_status"] = scored.t2_status
        result["spe_status"] = scored.spe_status
        result["overall_status"] = scored.overall_status
        result["score_valid"] = scored.score_valid
        result["invalid_reason"] = scored.invalid_reason
        result["status"] = result["overall_status"]
        return result


def fit_dpca(
    frame: pd.DataFrame,
    variance_threshold: float = 0.95,
    n_components: int | None = None,
) -> DPCAModel:
    if frame.columns.has_duplicates:
        raise ValueError("feature names must be unique")
    if not all(isinstance(name, str) for name in frame.columns):
        raise ValueError("feature names must be strings")
    values = frame.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("DPCA requires at least three rows and two features")
    if not np.isfinite(values).all():
        raise ValueError("training data contain non-finite values")
    if not 0 < variance_threshold < 1:
        raise ValueError(
            "variance threshold must be in (0, 1) to preserve residual space for SPE"
        )

    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("constant or near-constant features cannot be standardized")
    standardized = (values - mean) / scale

    max_components = min(values.shape[0] - 1, values.shape[1])
    full_pca = PCA(n_components=max_components, svd_solver="full")
    full_pca.fit(standardized)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    effective_rank = int(
        np.sum(full_pca.explained_variance_ > np.finfo(float).eps)
    )
    if effective_rank < 3:
        raise ValueError(
            "DPCA effective rank must be at least 3 to provide PC1, PC2, "
            "and effective residual space for SPE"
        )

    if n_components is None:
        selected = max(
            2, int(np.searchsorted(cumulative, variance_threshold) + 1)
        )
    else:
        selected = int(n_components)
    if not 2 <= selected <= max_components:
        raise ValueError(f"n_components must be between 2 and {max_components}")
    if selected >= effective_rank:
        raise ValueError(
            "selected components leave no effective residual space for SPE; "
            "use fewer components or provide richer training data"
        )

    eigenvalues = full_pca.explained_variance_.copy()
    t2_limits = {
        alpha: _t2_limit(values.shape[0], selected, alpha)
        for alpha in (0.95, 0.99)
    }
    residual_eigenvalues = eigenvalues[selected:]
    training_scores = standardized @ full_pca.components_[:selected].T
    training_residual = standardized - training_scores @ full_pca.components_[:selected]
    training_spe = np.sum(training_residual**2, axis=1)
    q_limits = {
        alpha: _q_limit(residual_eigenvalues, training_spe, alpha)
        for alpha in (0.95, 0.99)
    }

    return DPCAModel(
        feature_names=tuple(frame.columns),
        mean=mean,
        scale=scale,
        components=full_pca.components_[:selected].copy(),
        eigenvalues=eigenvalues,
        explained_variance_ratio=full_pca.explained_variance_ratio_.copy(),
        t2_limits=t2_limits,
        q_limits=q_limits,
        n_samples=values.shape[0],
    )


def _t2_limit(n_samples: int, n_components: int, alpha: float) -> float:
    multiplier = (
        n_components
        * (n_samples - 1)
        * (n_samples + 1)
        / (n_samples * (n_samples - n_components))
    )
    return float(multiplier * f.ppf(alpha, n_components, n_samples - n_components))


def _q_limit(
    residual_eigenvalues: np.ndarray,
    training_spe: np.ndarray,
    alpha: float,
) -> float:
    if residual_eigenvalues.size == 0 or np.all(residual_eigenvalues <= 0):
        return 0.0
    theta1 = float(np.sum(residual_eigenvalues))
    theta2 = float(np.sum(residual_eigenvalues**2))
    theta3 = float(np.sum(residual_eigenvalues**3))
    if theta1 <= 0 or theta2 <= 0:
        return float(np.quantile(training_spe, alpha))

    h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2**2)
    if h0 <= 0:
        return float(np.quantile(training_spe, alpha))
    bracket = (
        norm.ppf(alpha) * np.sqrt(2.0 * theta2 * h0**2) / theta1
        + 1.0
        + theta2 * h0 * (h0 - 1.0) / theta1**2
    )
    if bracket <= 0:
        return float(np.quantile(training_spe, alpha))
    return float(theta1 * bracket ** (1.0 / h0))
