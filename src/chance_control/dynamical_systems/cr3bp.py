import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import (AbstractSolver, AbstractStepSizeController,
                     ConstantStepSize, Dopri8, PIDController)
from jaxtyping import Array, Float, Key, jaxtyped

from chance_control.dynamical_systems import AbstractContinuousDynamicalSystem


@jaxtyped(typechecker=typechecker)
class CR3BP(AbstractContinuousDynamicalSystem, strict=True):
    ### Dynamical System Parameters
    # From JPL CR3BP Earth-Moon Mass Ratio
    mu: float = 0.01215058560962404

    # Mean is likely to change to be initial points from CR3BP library
    # mean: Float[Array, "6"] = eqx.field(
    #     default_factory=lambda: jnp.array([0.928198691381327, 0.107788360285704, 0.341251149344347, 0.122057469025560, 0.00824091751033246, 0.312754417877696])
    # )
    mean: Float[Array, "6"] = eqx.field(
        default_factory=lambda: jnp.array(
            [
                1.03023361221087,
                0.0687372443157935,
                0.147641949569388,
                0.157623720622668,
                -0.181091773654881,
                0.389575654309805,
            ]
        )
    )
    # Covariance is from
    # Efficient Orbit Determination Using Measurement-Directional State Transition Tensor
    covariance: Float[Array, "6 6"] = eqx.field(
        default_factory=lambda: jnp.block(
            [
                [(2.5e-5) ** 2 * jnp.eye(3), jnp.zeros((3, 3))],
                [jnp.zeros((3, 3)), (1e-6) ** 2 * jnp.eye(3)],
            ]
        )
    )

    ### Solver Parameters
    dt: float = 0.0001
    solver: AbstractSolver = Dopri8()
    stepsize_controller: AbstractStepSizeController = PIDController(
        rtol=1e-12, atol=1e-14
    )

    @property
    def dimension(self):
        return 6

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        if key is None:
            return self.mean
        else:
            return jax.random.multivariate_normal(
                key, mean=self.mean, cov=self.covariance
            )

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def vector_field(self, t, y, args):
        r, v = y[:3], y[3:6]
        re = jnp.sqrt((r[0] + self.mu) ** 2 + r[1] ** 2 + r[2] ** 2)
        rm = jnp.sqrt((r[0] - 1 + self.mu) ** 2 + r[1] ** 2 + r[2] ** 2)
        assert re.shape == () and rm.shape == ()
        return jnp.concatenate(
            [
                v,
                jnp.array(
                    [
                        r[0]
                        + 2 * v[1]
                        - (1 - self.mu) * (r[0] + self.mu) / re**3
                        - self.mu * (r[0] - 1 + self.mu) / rm**3,
                        r[1]
                        - 2 * v[0]
                        - (1 - self.mu) * r[1] / re**3
                        - self.mu * r[1] / rm**3,
                        -(1 - self.mu) * r[2] / re**3 - self.mu * r[2] / rm**3,
                    ]
                ),
            ]
        )
