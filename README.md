# Gillespie
Numba accelerated implementation of Gillespie's algorithm for simulating stochastic processes. This implementation was inspiredd by the following [project](https://github.com/wefatherley/monte-carlo).

This implementation is around 6 times faster but less user-friendly.

# Usage
Within gillespie.py, gillespie_direct() is the actual implementation of Gillespie's algorithm. The remaining of the code is an example of its usage on the SIR model as explained on this [wikipedia] (https://en.wikipedia.org/wiki/Gillespie_algorithm) page.

To adapt this code to your situation, the following must be changed: data, stoichiometry and propensity()

# References
Exact stochastic simulation of coupled chemical reactions : https://pubs.acs.org/doi/abs/10.1021/j100540a008

Modéliser la propagation d’une épidémie : http://www.math.ens.fr/enseignement/telecharger_fichier.php?fichier=1693
