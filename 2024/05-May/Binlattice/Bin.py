
import numpy as np

def option_price_bin_policy(spot_price: float, strike: float, expiry: float, rate: float, vol: float, num_steps: int) -> float:
    """
    This function finds the price of an option using the Binomial lattice.
    The underlying price follows a Geometric Brownian Motion (GBM).
    """
    # Number of time steps
    dt = expiry / num_steps

    # Up factor
    u = np.exp(vol * np.sqrt(dt))

    # Down factor
    d = 1 / u

    # Probability of upward movement
    q = (np.exp(rate * dt) - d) / (u - d)

    # Initialize the option price dictionary
    C = {}

    # Calculate payoff for put options at maturity
    for m in range(0, num_steps + 1):
        C[(num_steps, m)] = max(strike - spot_price * (u ** (2 * m - num_steps)), 0)

    # Backward induction to calculate option price at each node
    for k in range(num_steps - 1, -1, -1):
        for m in range(0, k + 1):
            future_value = np.exp(-rate * dt) * (q * C[(k + 1, m + 1)] + (1 - q) * C[(k + 1, m)])
            exercise_value = max(strike - spot_price * (u ** (2 * m - k)), 0)
            C[(k, m)] = max(future_value, exercise_value)

    return C[(0, 0)]

