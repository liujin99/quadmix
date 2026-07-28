"""
Parameter sampling for QuaDMix — Algorithm 1 from the paper.

The algorithm generates random QuaDMix parameter configurations:
  1. Sample global criteria weights (a₁, ..., aₙ) ~ U(0,1), normalize
  2. For each domain m:
     a. Sample domain-specific weights (b_{1,m}, ..., b_{N,m}) ~ U(0,1)
     b. α_m = normalized product of global and domain weights
     c. Sample (λ, ω, η, ε) ~ U(0,1), rescale to meaningful ranges
     d. β_m = (λ̃, ω̃, η̃, ε̃)
  3. θ = (θ₁, ..., θ_M) where θ_m = (α_m, β_m)
"""

import numpy as np
import numpy.typing as npt
from typing import List, Optional

from quadmix.core.types import (
    MergedQualityConfig,
    SamplingConfig,
    ParameterSet,
    QuaDMixConfig,
)


class ParameterSampler:
    """
    Generates QuaDMix parameter configurations (Algorithm 1).

    Produces θ_i = (θ_1, ..., θ_M) where each θ_m = (α_m, β_m).
    Total parameters per configuration: (N + 4) × M.
    """

    def __init__(self, config: QuaDMixConfig, seed: Optional[int] = None):
        self.config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def sample_one(self) -> ParameterSet:
        """
        Generate a single parameter configuration (Algorithm 1).

        Returns:
            A ParameterSet with randomly sampled parameters.
        """
        N = self.config.num_criteria
        M = self.config.num_domains

        # === Step 1: Sample global criteria weights ===
        # Sample (a₁, ..., aₙ) ~ U(0, 1)
        a = self._rng.uniform(0, 1, size=N)
        # Normalize: ãₙ = aₙ / Σ aᵢ
        a_norm = a / max(a.sum(), 1e-10)

        # === Step 2: For each domain m ===
        domain_weights_list = []
        sampling_configs = []

        for m in range(M):
            # Sample domain-specific weights (b₁, ..., bₙ) ~ U(0, 1)
            b = self._rng.uniform(0, 1, size=N)

            # Compute final weight: b̃ₙ = ãₙ · bₙ / Σ ãᵢ · bᵢ
            b_raw = a_norm * b
            b_norm = b_raw / max(b_raw.sum(), 1e-10)

            # α_m = (b̃₁, ..., b̃ₙ)
            domain_weights_list.extend(b_norm)

            # Sample (λ, ω, η, ε) ~ U(0, 1)
            lambda_raw = self._rng.uniform(self.config.lambda_min, self.config.lambda_max)
            omega_raw = self._rng.uniform(self.config.omega_min, self.config.omega_max)
            eta_raw = self._rng.uniform(self.config.eta_min, self.config.eta_max)
            epsilon_raw = self._rng.uniform(self.config.epsilon_min, self.config.epsilon_max)

            # Rescale to meaningful ranges
            # λ̃ = 10³·λ — controls how fast sampling decreases
            # ω̃ = 0.1·ω — quality threshold (0 to 0.1 means top 10% gets boost)
            # η̃ = η — scaling factor
            # ε̃ = ε/1000 — very small base rate
            rescaled_lambda = lambda_raw * self.config.lambda_scale
            rescaled_omega = omega_raw * self.config.omega_scale
            rescaled_eta = eta_raw * self.config.eta_scale
            rescaled_epsilon = epsilon_raw * self.config.epsilon_scale

            sampling_configs.append(SamplingConfig(
                lambda_=rescaled_lambda,
                omega=rescaled_omega,
                eta=rescaled_eta,
                epsilon=rescaled_epsilon,
            ))

        # global_weights stores the sampled ã for reference/debugging only.
        # It is an intermediate value in Algorithm 1, NOT part of the
        # optimizable parameter vector θ. flatten/unflatten ignores it.
        merge_config = MergedQualityConfig(
            global_weights=np.array(a_norm, dtype=np.float64),
            domain_weights=np.array(domain_weights_list, dtype=np.float64),
        )

        return ParameterSet(
            merge_config=merge_config,
            sampling_configs=sampling_configs,
        )

    def _draw_raw_uniform(self, n: int, N: int, M: int):
        """Draw raw samples via i.i.d. uniform (legacy path)."""
        a_all = self._rng.uniform(0, 1, size=(n, N))
        b_all = self._rng.uniform(0, 1, size=(n, M, N))
        lambda_all = self._rng.uniform(self.config.lambda_min, self.config.lambda_max, size=(n, M)) * self.config.lambda_scale
        omega_all = self._rng.uniform(self.config.omega_min, self.config.omega_max, size=(n, M)) * self.config.omega_scale
        eta_all = self._rng.uniform(self.config.eta_min, self.config.eta_max, size=(n, M)) * self.config.eta_scale
        epsilon_all = self._rng.uniform(self.config.epsilon_min, self.config.epsilon_max, size=(n, M)) * self.config.epsilon_scale
        return a_all, b_all, lambda_all, omega_all, eta_all, epsilon_all

    def _draw_raw_sobol(self, n: int, N: int, M: int):
        """Draw raw samples via scrambled Sobol sequence (low-discrepancy).

        Layout of the d = N + M*(N+4) Sobol columns:
            [ a(N) | b(M*N) | lambda(M) | omega(M) | eta(M) | epsilon(M) ]
        Each slice is mapped from [0,1) to its configured [min,max] range and
        rescaled, exactly mirroring _draw_raw_uniform's per-group transform.
        """
        from scipy.stats import qmc
        d = N + M * (N + 4)
        eng = qmc.Sobol(d=d, scramble=True, seed=self._seed)
        u = eng.random(n)
        a_all = u[:, :N]
        b_all = u[:, N:N + M * N].reshape(n, M, N)
        c = N + M * N
        cfg = self.config
        lambda_all = (cfg.lambda_min + u[:, c:c + M] * (cfg.lambda_max - cfg.lambda_min)) * cfg.lambda_scale
        omega_all = (cfg.omega_min + u[:, c + M:c + 2 * M] * (cfg.omega_max - cfg.omega_min)) * cfg.omega_scale
        eta_all = (cfg.eta_min + u[:, c + 2 * M:c + 3 * M] * (cfg.eta_max - cfg.eta_min)) * cfg.eta_scale
        epsilon_all = (cfg.epsilon_min + u[:, c + 3 * M:c + 4 * M] * (cfg.epsilon_max - cfg.epsilon_min)) * cfg.epsilon_scale
        return a_all, b_all, lambda_all, omega_all, eta_all, epsilon_all

    def sample_batch(self, n: int) -> List[ParameterSet]:
        """
        Generate n parameter configurations (vectorized).

        Args:
            n: Number of parameter configurations to generate.

        Returns:
            List of n ParameterSet instances.
        """
        N = self.config.num_criteria
        M = self.config.num_domains

        method = getattr(self.config, "sampler_method", "sobol")
        if method == "sobol":
            a_all, b_all, lambda_all, omega_all, eta_all, epsilon_all = self._draw_raw_sobol(n, N, M)
        elif method == "uniform":
            a_all, b_all, lambda_all, omega_all, eta_all, epsilon_all = self._draw_raw_uniform(n, N, M)
        else:
            raise ValueError(f"unknown sampler_method: {method!r}")

        a_norm_all = a_all / np.clip(a_all.sum(axis=1, keepdims=True), 1e-10, None)

        b_raw_all = a_norm_all[:, np.newaxis, :] * b_all
        b_norm_all = b_raw_all / np.clip(b_raw_all.sum(axis=2, keepdims=True), 1e-10, None)

        results = []
        for i in range(n):
            domain_weights = b_norm_all[i].flatten()
            sampling_configs = [
                SamplingConfig(
                    lambda_=lambda_all[i, m],
                    omega=omega_all[i, m],
                    eta=eta_all[i, m],
                    epsilon=epsilon_all[i, m],
                )
                for m in range(M)
            ]
            merge_config = MergedQualityConfig(
                global_weights=a_norm_all[i],
                domain_weights=domain_weights,
            )
            results.append(ParameterSet(merge_config=merge_config, sampling_configs=sampling_configs))

        return results
