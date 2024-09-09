
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \text{reward} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$


$$
\underline{\text{td\_target}} = \text{reward} + \gamma \max_{a'} Q(s', a')
$$

$$
\underline{\text{td\_error}} = \underline{\text{td\_target}} - Q(s, a)
$$

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( \underline{\text{td\_error}} \right)
$$
