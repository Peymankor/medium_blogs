import numpy as np

def option_price_binomial(initial_stock_price: float, strike_price: float, expiry_time: float, risk_free_rate: float, volatility: float, num_time_steps: int) -> float:
    """
    This function calculates the price of an option using the Binomial lattice model.
    The underlying asset price follows a Geometric Brownian Motion (GBM).
    """
    # Time step
    dt = expiry_time / num_time_steps

    # Up factor
    u = np.exp(volatility * np.sqrt(dt))

    # Down factor
    d = 1 / u

    # Probability of upward movement
    q = (np.exp(risk_free_rate * dt) - d) / (u - d)

    # Initialize the option price dictionary
    option_prices = {}

    # Calculate payoff for put options at maturity
    for m in range(num_time_steps + 1):
        option_prices[(num_time_steps, m)] = max(strike_price - initial_stock_price * (u ** (2 * m - num_time_steps)), 0)

    # Backward induction for option pricing
    for k in range(num_time_steps - 1, -1, -1):
        for m in range(k + 1):
            future_value = np.exp(-risk_free_rate * dt) * (q * option_prices[(k + 1, m + 1)] + (1 - q) * option_prices[(k + 1, m)])
            exercise_value = max(strike_price - initial_stock_price * (u ** (2 * m - k)), 0)
            option_prices[(k, m)] = max(future_value, exercise_value)

    return option_prices[(0, 0)]



# Example usage
spot_price_val = 36
strike_val = 40
expiry_val = 1
rate_val = 0.06
vol_val = 0.2
num_steps_val = 50

option_price_BOPM = option_price_binomial(spot_price=spot_price_val, 
                                strike=strike_val, 
                                expiry=expiry_val, 
                                rate=rate_val, 
                                vol=vol_val, 
                                num_steps=num_steps_val)

print("Option Price using BOPM Method: ", option_price_BOPM)