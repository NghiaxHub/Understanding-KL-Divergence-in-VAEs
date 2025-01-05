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
