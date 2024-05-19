#from rich import pretty
#pretty.install()

prior_prob_rain = 20/30

print(f"Prior Probaility of Rain: {prior_prob_rain}")

prob_heavycloud_rain = 0.7
prob_heavycloud_norain = 0.3

# Calculate the total probability of heavy cloud
prob_heavycloud = prior_prob_rain * prob_heavycloud_rain + (1 - prior_prob_rain) * prob_heavycloud_norain

# Calculate the updated probability of rain
updated_prob_rain = prior_prob_rain * prob_heavycloud_rain / prob_heavycloud

# Print the results
#print(f"Total Probability of Heavy Cloud: {prob_heavycloud}")
print(f"Updated Probability of Rain: {updated_prob_rain}")



