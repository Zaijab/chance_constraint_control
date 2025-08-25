"""
Gaussian Mixture Model (GMM) framework for chance-constrained control.

This module implements GMM representations and propagation as described in
"Chance-Constrained Gaussian Mixture Steering to a Terminal Gaussian Distribution"
(Kumagai & Oguri, 2024).

Key theoretical results implemented:
- Proposition 3: State distribution under proposed control policy
- Proposition 4: Control distribution  
- Proposition 5: Terminal constraint sufficient condition
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Shaped, jaxtyped


class GaussianMixtureModel(eqx.Module, strict=True):
    """
    Gaussian Mixture Model representation.
    
    A GMM is characterized by:
    - weights: α_i for i = 1, ..., K (sum to 1)
    - means: μ_i for i = 1, ..., K  
    - covariances: Σ_i for i = 1, ..., K
    """
    
    weights: Float[Array, "num_components"]
    means: Float[Array, "num_components dim"]
    covariances: Float[Array, "num_components dim dim"]
    
    def __init__(
        self,
        weights: Float[Array, "num_components"],
        means: Float[Array, "num_components dim"],
        covariances: Float[Array, "num_components dim dim"],
    ):
        """
        Initialize a Gaussian Mixture Model.
        
        Args:
            weights: Component weights (must sum to 1)
            means: Component means
            covariances: Component covariances (must be positive semi-definite)
        """
        # Validate inputs
        assert jnp.allclose(jnp.sum(weights), 1.0), "Weights must sum to 1"
        assert jnp.all(weights >= 0), "Weights must be non-negative"
        assert means.shape[0] == weights.shape[0], "Number of means must match weights"
        assert covariances.shape[0] == weights.shape[0], "Number of covariances must match weights"
        assert covariances.shape[1] == covariances.shape[2], "Covariances must be square"
        assert means.shape[1] == covariances.shape[1], "Mean and covariance dimensions must match"
        
        self.weights = weights
        self.means = means
        self.covariances = covariances
    
    @property
    def num_components(self) -> int:
        """Number of mixture components."""
        return self.weights.shape[0]
    
    @property
    def dimension(self) -> int:
        """Dimension of the random variable."""
        return self.means.shape[1]
    
    @jaxtyped(typechecker=typechecker)
    def overall_mean(self) -> Float[Array, "dim"]:
        """
        Compute overall mean: μ^g = Σ_i α_i * μ_i
        
        Returns:
            Overall mean of the mixture
        """
        return jnp.sum(self.weights[:, None] * self.means, axis=0)
    
    @jaxtyped(typechecker=typechecker)
    def overall_covariance(self) -> Float[Array, "dim dim"]:
        """
        Compute overall covariance: Σ^g = Σ_i α_i * (Σ_i + μ_i * μ_i^T) - μ^g * (μ^g)^T
        
        Returns:
            Overall covariance of the mixture
        """
        mu_g = self.overall_mean()
        
        # Compute weighted sum of (covariance + outer product of means)
        weighted_second_moments = jnp.sum(
            self.weights[:, None, None] * (
                self.covariances + 
                self.means[:, :, None] @ self.means[:, None, :]
            ),
            axis=0
        )
        
        return weighted_second_moments - mu_g[:, None] @ mu_g[None, :]
    
    @jaxtyped(typechecker=typechecker)
    def log_prob(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        """
        Compute log probability density at point x.
        
        Args:
            x: Point to evaluate
            
        Returns:
            Log probability density
        """
        # Compute log probabilities for each component
        def component_log_prob(i):
            return jsp.stats.multivariate_normal.logpdf(
                x, self.means[i], self.covariances[i]
            )
        
        log_probs = jnp.array([component_log_prob(i) for i in range(self.num_components)])
        
        # Use logsumexp for numerical stability
        return jsp.special.logsumexp(jnp.log(self.weights) + log_probs)
    
    @jaxtyped(typechecker=typechecker)
    def sample(
        self, 
        key: Array, 
        num_samples: int = 1
    ) -> Float[Array, "num_samples dim"]:
        """
        Sample from the mixture model.
        
        Args:
            key: Random key
            num_samples: Number of samples to generate
            
        Returns:
            Samples from the mixture
        """
        key_component, key_gaussian = jax.random.split(key)
        
        # Sample component indices
        component_indices = jax.random.choice(
            key_component, 
            self.num_components, 
            shape=(num_samples,), 
            p=self.weights
        )
        
        # Sample from selected components
        def sample_from_component(i, subkey):
            return jax.random.multivariate_normal(
                subkey, self.means[i], self.covariances[i]
            )
        
        keys = jax.random.split(key_gaussian, num_samples)
        samples = jnp.array([
            sample_from_component(component_indices[i], keys[i])
            for i in range(num_samples)
        ])
        
        return samples if num_samples > 1 else samples[0]


@jaxtyped(typechecker=typechecker)
def posterior_weights(
    x0: Float[Array, "dim"],
    prior_gmm: GaussianMixtureModel,
) -> Float[Array, "num_components"]:
    """
    Compute posterior weights λ_i(x_0) given initial state observation.
    
    From paper: λ_i(x_0) = α_i * f_N(x_0; μ_i^0, Σ_i^0) / Σ_j α_j * f_N(x_0; μ_j^0, Σ_j^0)
    
    Args:
        x0: Observed initial state
        prior_gmm: Prior GMM distribution
        
    Returns:
        Posterior weights for each component
    """
    # Compute likelihood for each component
    def component_likelihood(i):
        return jsp.stats.multivariate_normal.pdf(
            x0, prior_gmm.means[i], prior_gmm.covariances[i]
        )
    
    likelihoods = jnp.array([
        component_likelihood(i) for i in range(prior_gmm.num_components)
    ])
    
    # Compute posterior weights
    numerator = prior_gmm.weights * likelihoods
    denominator = jnp.sum(numerator)
    
    return numerator / denominator


@jaxtyped(typechecker=typechecker)
def propagate_gmm_state(
    gmm: GaussianMixtureModel,
    A: Float[Array, "out_dim in_dim"],
    B: Float[Array, "out_dim control_dim"],
    V: Float[Array, "control_dim"],
    L: Float[Array, "num_components control_dim in_dim"],
) -> GaussianMixtureModel:
    """
    Propagate GMM through linear dynamics with affine control policy.
    
    Implements Proposition 3 from the paper:
    x_{k+1} follows GMM with:
    - Same weights α_i
    - Means: μ_i^{k+1} = A * μ_i^k + B * V + B * L^i * (μ_i^k - μ^g)
    - Covariances: Σ_i^{k+1} = (A + B * L^i) * Σ_i^k * (A + B * L^i)^T
    
    Args:
        gmm: Current GMM distribution
        A: State transition matrix  
        B: Control input matrix
        V: Feedforward control term
        L: Feedback gains for each component
        
    Returns:
        Propagated GMM distribution
    """
    mu_g = gmm.overall_mean()
    num_components = gmm.num_components
    
    # Propagate each component
    new_means = []
    new_covariances = []
    
    for i in range(num_components):
        # Mean propagation: μ_i^{k+1} = A * μ_i^k + B * V + B * L^i * (μ_i^k - μ^g)
        mu_i = gmm.means[i]
        L_i = L[i]
        
        new_mean = A @ mu_i + B @ V + B @ L_i @ (mu_i - mu_g)
        new_means.append(new_mean)
        
        # Covariance propagation: Σ_i^{k+1} = (A + B * L^i) * Σ_i^k * (A + B * L^i)^T
        Sigma_i = gmm.covariances[i]
        A_tilde = A + B @ L_i
        
        new_cov = A_tilde @ Sigma_i @ A_tilde.T
        new_covariances.append(new_cov)
    
    new_means_array = jnp.stack(new_means)
    new_covariances_array = jnp.stack(new_covariances)
    
    return GaussianMixtureModel(
        weights=gmm.weights,
        means=new_means_array,
        covariances=new_covariances_array
    )


@jaxtyped(typechecker=typechecker)
def propagate_gmm_control(
    gmm: GaussianMixtureModel,
    V: Float[Array, "control_dim"],
    L: Float[Array, "num_components control_dim state_dim"],
) -> GaussianMixtureModel:
    """
    Compute control distribution under affine policy.
    
    Implements Proposition 4 from the paper:
    u_k follows GMM with:
    - Same weights α_i
    - Means: μ_{u,i}^k = v_k + L_k^i * (μ_i^0 - μ^g)
    - Covariances: Σ_{u,i}^k = L_k^i * Σ_i^0 * (L_k^i)^T
    
    Args:
        gmm: Initial state GMM distribution (at time 0)
        V: Feedforward control term
        L: Feedback gains for each component
        
    Returns:
        Control GMM distribution
    """
    mu_g = gmm.overall_mean()
    num_components = gmm.num_components
    
    new_means = []
    new_covariances = []
    
    for i in range(num_components):
        mu_i = gmm.means[i]
        Sigma_i = gmm.covariances[i]
        L_i = L[i]
        
        # Control mean: μ_{u,i}^k = v_k + L_k^i * (μ_i^0 - μ^g)
        new_mean = V + L_i @ (mu_i - mu_g)
        new_means.append(new_mean)
        
        # Control covariance: Σ_{u,i}^k = L_k^i * Σ_i^0 * (L_k^i)^T
        new_cov = L_i @ Sigma_i @ L_i.T
        new_covariances.append(new_cov)
    
    new_means_array = jnp.stack(new_means)
    new_covariances_array = jnp.stack(new_covariances)
    
    return GaussianMixtureModel(
        weights=gmm.weights,
        means=new_means_array,
        covariances=new_covariances_array
    )


@jaxtyped(typechecker=typechecker)
def terminal_constraint_matrices(
    terminal_gmm: GaussianMixtureModel,
    target_covariance: Float[Array, "dim dim"],
) -> Float[Array, "dim total_cov_dim"]:
    """
    Compute matrix Y for terminal constraint (Proposition 6).
    
    Terminal constraint: ||Σ_f^{-1/2} * Y||_2 ≤ 1
    where Y = [√α_1 * (Σ_1^N)^{1/2}, ..., √α_K * (Σ_K^N)^{1/2}]
    
    Args:
        terminal_gmm: Terminal GMM distribution
        target_covariance: Target covariance Σ_f
        
    Returns:
        Matrix Y for constraint formulation
    """
    # Compute matrix square roots of component covariances
    component_sqrt_covs = []
    
    for i in range(terminal_gmm.num_components):
        Sigma_i = terminal_gmm.covariances[i]
        # Compute matrix square root via eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(Sigma_i)
        sqrt_Sigma_i = eigenvecs @ jnp.diag(jnp.sqrt(jnp.maximum(eigenvals, 0))) @ eigenvecs.T
        
        weighted_sqrt_cov = jnp.sqrt(terminal_gmm.weights[i]) * sqrt_Sigma_i
        component_sqrt_covs.append(weighted_sqrt_cov)
    
    # Concatenate horizontally
    Y = jnp.concatenate(component_sqrt_covs, axis=1)
    
    return Y


@jaxtyped(typechecker=typechecker)
def create_paper_initial_gmm() -> GaussianMixtureModel:
    """
    Create the initial GMM distribution from the paper's numerical example.
    
    3 components with:
    - weights: (0.3, 0.4, 0.3)
    - means: μ^(1) = [5, -1, 5, 0]^T, μ^(2) = [3.5, 0.5, 8, 0]^T, μ^(3) = [4, -0.5, 7, 0]^T
    - covariances: Σ^(i) = diag(0.05, 0.05, 0.01, 0.01) for all i
    
    Returns:
        Initial GMM from the paper
    """
    weights = jnp.array([0.3, 0.4, 0.3])
    
    means = jnp.array([
        [5.0, -1.0, 5.0, 0.0],
        [3.5, 0.5, 8.0, 0.0],
        [4.0, -0.5, 7.0, 0.0]
    ])
    
    # Same covariance for all components
    base_cov = jnp.diag(jnp.array([0.05, 0.05, 0.01, 0.01]))
    covariances = jnp.stack([base_cov, base_cov, base_cov])
    
    return GaussianMixtureModel(weights, means, covariances)


@jaxtyped(typechecker=typechecker)
def create_paper_target_gaussian() -> tuple[Float[Array, "4"], Float[Array, "4 4"]]:
    """
    Create the target terminal Gaussian distribution from the paper.
    
    Target: μ_f = [8, 5.5, 0, 0]^T, Σ_f = diag(0.05, 0.05, 0.01, 0.01)
    
    Returns:
        (mean, covariance) of target distribution
    """
    mean = jnp.array([8.0, 5.5, 0.0, 0.0])
    covariance = jnp.diag(jnp.array([0.05, 0.05, 0.01, 0.01]))
    
    return mean, covariance