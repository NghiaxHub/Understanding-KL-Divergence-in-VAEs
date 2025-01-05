# Kullback-Leibler Divergence in VAEs

## 1. Target

Given the dataset $D$, with $p^*$ is the real distribution of $D$, where $x$ is sampled from the dataset $D$: $x \sim D$.

We want to build some probabilistic model $p$ parameterized by $\theta$ that will approximate $p^*$ as closely as possible.

$$
p^*(x) \approx p_{\theta}(x)
$$

where $p_{\theta}(x)$ is called the likelihood of the data $x$ given the model parameters $\theta$.

**Target:** Adjust $\theta$ to achieve the maximum likelihood of observing the data points actually included in our dataset $D$.

We want to use latent variables $z$ which can represent structure of original data $x$ and data representation of $x$ is just their shadow.

How do we find an efficient latent representation $z$ that captures the factors of variation inherent in the data $x$?

$$
{x \in D} \hspace{0.2cm} \overset{p_{\theta}(z|x)}{\underset{ \text{Inference}}{\xrightarrow{\hspace{1cm}}}} \hspace{0.2cm} {p_{\theta}(z)} \hspace{0.2cm} \overset{p_{\theta}(x|z)}{\underset{ \text{Generation}}{\xrightarrow{\hspace{1cm}}}} \hspace{0.2cm} {x \in D}
$$

All we have to do now is optimizing $\theta$ such that VAEs could be encouraged to map the full data distribution $D$ to a latent distribution $p_{\theta}(z)$, so if we choose a unit Gaussian as our $p_{\theta}(z)$ and sampling from this unit Gaussian, will produce realistic synthetic examples in data space.

Our **original target** is maximizing the model evidence $p_{\theta}(x)$:

$$
p_{\theta}(x) = \int_z p_{\theta}(z, x)dz = \int_z p_{\theta}(z) \cdot p_{\theta}(x | z)dz
$$

Because $p_{\theta}(x)$ only be found by integrating the joint probability distribution $p_{\theta}(z, x)$ over all possible values of $z$, that's why we call it the marginal likelihood of $x$.

But this integral is intractable (infeasible to compute) because it has no analytic solution and can not be efficiently estimated.

## 2. Solution to optimize $\theta$

We do not need to compute $p_{\theta}$ itself, we just need some way of increasing it as much as possible, and it does this within a Bayesian framework.

Bayes' rule for hypothesis $H$ and evidence $E$:

$$
p(H|E) = \frac{p(H) \cdot p(E|H)}{p(E)}
$$

Where:
- $p(H|E)$: Posterior probability
- $p(H)$: Prior probability
- $p(E|H)$: Likelihood of $E$ given $H$, measures the extent to which hypothesis $H$ explains the evidence $E$
- $p(E)$: Marginal likelihood or model evidence, regardless what $H$ is

Adapt the Bayes' rule to our situation (we use $p_{\theta}$ because we do not know the exact distribution of data $D$, we just approximate them by building probabilistic model over $\theta$):

$$
p_{\theta}(z|x) = \frac{p_{\theta}(z) \cdot p_{\theta}(x|z)}{p_{\theta}(x)}
$$

Where: model evidence $p_{\theta}(x)$ is the likelihood that our distribution parameters $\theta$ are appropriate the data $x$ that we observe.

To maximize the model evidence, we have to find some $\theta$ for which the right hand side (RHS) is maximized:

$$
p_{\theta}(x) = \frac{p_{\theta}(x, z)}{p_{\theta}(z|x)}
$$

Because the marginal likelihood $p_{\theta}(x)$ is intractable, the posterior $p_{\theta}(z|x)$ is also intractable since the marginal likelihood is required in the calculation of the posterior. Nevertheless, if we know one of them, we could compute the other, given that the joint probability $p_{\theta}(x,z)$ is tractable.

By using method of Variational Inference, we can approximate the posterior instead of directly computing the marginal likelihood $p_{\theta}(x)$:

$$
q_{\phi}(z|x) \approx p_{\theta}(z|x)
$$

where $p_{\theta}$ is the target distribution, the approximation $q$ is parameterized by $\phi$. This distribution $q$ should represent some sufficiently flexible family of distributions such that by optimizing $\phi$, we can push these two distributions as closely together as possible.

## 3. Kullback-Leibler Divergence

KL Divergence is defined as the expected information loss suffered when we adopt $Q$ as our model while the true distribution is $P$.

$$
D_{KL}(Q || P) = \int_z Q(z) \log\left(\frac{Q(z)}{P(z)}\right)dz
$$

and important property:

$$
D_{KL}(Q || P) \geq 0; \space D_{KL} = 0 \Leftrightarrow Q = P
$$

KL Divergence in our target to our approximation:

Let $q_{\phi} = q_{\phi}(z|x), \space p_{\theta} = p_{\theta}(z|x)$. Then we have:

$$
D_{KL}(q_{\phi} || p_{\theta}) = \int_z q_{\phi} \cdot \log \left(\frac{q_{\phi}}{p_{\theta}}\right)dz
$$

By applying definition of the expected value for a distribution, we have:

$$
D_{KL}(q_{\phi} || p_{\theta}) = \mathbb{E}_{q_{\phi}}\left[\log\left(\frac{q_{\phi}}{p_{\theta}}\right)\right] 
$$
$$
= \mathbb{E}_{q_{\phi}}\left[\log q_{\phi}\right] - \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}\right]
$$
$$
= \mathbb{E}_{q_{\phi}}\left[\log q_{\phi}\right] - \mathbb{E}_{q_{\phi}}\left[\log \left( \frac{p_{\theta}(z, x)}{p_{\theta}(x)} \right)\right]
$$
$$
= \mathbb{E}_{q_{\phi}}\left[\log q_{\phi}\right] - \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(z, x)\right] + \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(x)\right]
$$
$$
= \log p_{\theta}(x) - \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(z,x) - \log q_{\phi}\right]
$$

In the last line, the first term does not have the expectation sign since $p_{\theta}(x)$ is independent of $z$.

The KL Divergence we have extracted can not be computed directly since the marginal likelihood $p_{\theta}(x)$. But this formula can help us deal with the intractability of $p_{\theta}(x)$.

We reshuffle the final formula:

$$
\log p_{\theta}(x) = D_{KL}(q_{\phi} || p_{\theta}) + \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(z,x) - \log q_{\phi}\right]
$$

as we have mentioned above, the KL Divergence is never negative, so:

$$
\log p_{\theta}(x) \geq \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(z,x) - \log q_{\phi}\right]
$$

This extra term essentially sets a lower bound on the value of the log-likelihood. In the context $p_{\theta}(x)$ is called the model evidence, we called this extra term the **evidence lower bound** or **ELBO**.

We can alternatively derive the ELBO using Jensen's inequality:

$$
f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]
$$

We have:

$$
\begin{aligned}
\log p_{\theta}(x) &= \log \int_z p_{\theta}(z, x)dz\\
&= \log \int_z p_{\theta}(z, x) \frac{q_{\phi}}{q_{\phi}}dz\\
&= \log \left(\mathbb{E}\left[ \frac{p_{\theta}(z, x)}{q_{\phi}}\right]\right)\\
&\geq \mathbb{E}_{q_{\phi}}\left[\log p_{\theta}(z,x) - \log q_{\phi}\right]
\end{aligned}
$$

By using this result, we can think KL Divergence from $p_{\theta}$ to $q_{\phi}$ is the gap between the ELBO and the actual log-likelihood $\log p_{\theta}(x)$. A lower KL Divergence increases the tightness of this bound.

```
^
|
|----------------------------------- log p_θ(x)
|         ∧
|         |
|         |
|         |    D_KL(q_φ||p_θ)
|         |
|         |
|         ∨
|----------------------------------- 𝔼_q_φ[log p_θ(z,x)/q_φ]
```

As we can see from this picture that maximizing the ELBO by optimizing the $\phi$ and $\theta$ will simultaneously do two things:

1. Maximize the $\log p_{\theta}(x)$
2. Minimize the KL Divergence from the true posterior $p_{\theta}$ to the approximation $q_{\phi}$
