import scipy.signal as sig
from pydrake.systems.primitives import LinearSystem


def transfer_function_to_linear_system(
    continuous_time_numer_coeffs: tuple[int | float, ...] = (1,),
    continuous_time_denom_coeffs: tuple[int | float, ...] = (1,),
    sample_period_s: float | None = None,
) -> LinearSystem:
    """
    Converts a continuous time transfer function to a Drake `LinearSystem` instance.

    Notes
    -----
    - If the argument `sample_period_s` is `None` (the default) or a real number less
      than or equal to zero, then the resulting
      `pydrake.systems.primitives.LinearSystem` instance is **continuous time**.
      Otherwise, if the `sample_period_s` is a positive real number, then the resulting
      `pydrake.systems.primitives.LinearSystem` instance is **discrete time** with the
      sample period given by the value of `sample_period_s`.
    - `pydrake.systems.primitives.LinearSystem` objects use a **state-space**
      representation (`transfer_function_to_linear_system` performs this conversion
      automatically).
    - This function will **fail** if the specified transfer function is not proper (
      the degree of the numerator <= degree of the denominator) because this would
      result in a non-causal system (the current output depends on future outputs).

    Parameters
    ----------
    continuous_time_numer_coeffs:
        The continuous time (s-domain) coefficients of each powers of s (including the
        power 0) for all terms in the numerator of the transfer function to convert,
        ordered by decreasing power of s (i.e. the coefficient of the highest power of s
        goes first).

    continuous_time_denom_coeffs:
        The continuous time (s-domain) coefficients of each powers of s (including the
        power 0) for all terms in the denominator of the transfer function to convert,
        ordered by decreasing power of s (i.e. the coefficient of the highest power of s
        goes first).

    sample_period_s:
        The optional sample period of the transfer function. If this argument is a real
        number greater than zero, then the output of this function is a discrete time
        `LinearSystem` with this given sample period (in seconds). Otherwise, the
        output of this function is a continuous time `LinearSystem`.

    Returns
    -------
    Either a continuous time or discrete time `pydrake.systems.primitives.LinearSystem`
    instance that represents the transfer function specified by the function arguments.

    Example
    -------
    To convert the transfer function :math:`H(s) = \dfrac{s}{s^2 + 2s + 1}`,
    to a Drake **continuous time** `LinearSystem`, use:

    >>> from mechae263C_helpers.drake import transfer_function_to_linear_system
    >>> H = transfer_function_to_linear_system(
    ...     continuous_time_numer_coeffs=(1, 0),
    ...     continuous_time_denom_coeffs=(1, 2, 1)
    >>> )

    To convert the same transfer function to a Drake **discrete time** `LinearSystem`
    with sample frequency of 100 Hz, use:

    >>> from mechae263C_helpers.drake import transfer_function_to_linear_system
    >>> H = transfer_function_to_linear_system(
    ...     continuous_time_numer_coeffs=(1, 0),
    ...     continuous_time_denom_coeffs=(1, 2, 1),
    ...     sample_period_s=1/100
    >>> )

    """
    if len(continuous_time_numer_coeffs) == 0:
        raise ValueError(
            "The argument `continuous_time_numer_coeffs` of function "
            "`transfer_function_to_linear_system` must be a tuple of at least one "
            f'number, but "{continuous_time_numer_coeffs}" was provided.'
        )

    if len(continuous_time_denom_coeffs) == 0:
        raise ValueError(
            "The argument `continuous_time_denom_coeffs` of function "
            "`transfer_function_to_linear_system` must be a tuple of at least one "
            f'number, but "{continuous_time_denom_coeffs}" was provided.'
        )

    if sample_period_s is not None and not isinstance(sample_period_s, (int, float)):
        try:
            sample_period_s = float(sample_period_s)
        except (TypeError, ValueError):
            raise ValueError(
                "The argument `sample_period_s` of function "
                "`transfer_function_to_linear_system` must be a `None`, a real number, "
                "or convertible to a real number via `float()`, but the value "
                f'"{sample_period_s}" was provided.'
            )

    if sample_period_s is None or sample_period_s <= 0:
        continuous_time_transfer_fn = sig.TransferFunction(
            continuous_time_numer_coeffs, continuous_time_denom_coeffs
        )
        ss_rep = continuous_time_transfer_fn.to_ss()
    else:
        continuous_time_transfer_fn = sig.lti(
            continuous_time_numer_coeffs, continuous_time_denom_coeffs
        )
        ss_rep = continuous_time_transfer_fn.to_discrete(
            dt=float(sample_period_s), method="backward_diff"
        ).to_ss()

    A, B, C, D = ss_rep.A, ss_rep.B, ss_rep.C, ss_rep.D
    sample_period_s = 0 if sample_period_s is None else sample_period_s
    return LinearSystem(A, B, C, D, time_period=sample_period_s)
