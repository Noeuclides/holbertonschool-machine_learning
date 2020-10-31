# 0x07. Bayesian Probability

## Description
What you should learn from this project:

* What is Bayesian Probability?
* What is Bayes’ rule and how do you use it?
* What is a base rate?
* What is a prior?
* What is a posterior?
* What is a likelihood?


---

### [0. Likelihood](./0-likelihood.py)
* You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, n patients take the drug and x patients develop severe side effects. You can assume that x follows a binomial distribution.

Write a function def likelihood(x, n, P): that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects.


### [1. Intersection](./1-intersection.py)
* Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr): that calculates the intersection of obtaining this data with the various hypothetical probabilities.


### [2. Marginal Probability](./2-marginal.py)
* Based on 1-intersection.py, write a function def marginal(x, n, P, Pr): that calculates the marginal probability of obtaining the data.


### [3. Posterior](./3-posterior.py)
* Based on 2-marginal.py, write a function def posterior(x, n, P, Pr): that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data.

---

## Author
* **Nicolas Martinez Machado** - [Noeuclides](https://github.com/Noeuclides)