import jax
from numpyro.distributions.constraints import Constraint, _SingletonConstraint


class _GreaterThanOrEqual(Constraint):
    """ """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x >= self.lower_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += "(lower_bound={})".format(self.lower_bound)
        return fmt_string

    def feasible_like(self, prototype):
        """

        Parameters
        ----------
        prototype :

        Returns
        -------

        """
        return jax.numpy.broadcast_to(
            self.lower_bound, jax.numpy.shape(prototype)
        )

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), dict())


class _Nonnegative(_GreaterThanOrEqual, _SingletonConstraint):
    """Constrain to non-negative reals"""

    def __init__(self):
        super().__init__(-0.0)


# constraint aliases
greater_than_or_equal = _GreaterThanOrEqual
nonnegative = _Nonnegative()
