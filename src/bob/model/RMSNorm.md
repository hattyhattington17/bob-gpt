# RMSNorm - Root Mean Square Normalization
- For a token representation $x_{b,j}  \in \mathbb{R}^{\mathtt{d\_model}}$, RMSNorm normalizes the vector by dividing by its root mean square size
- goal - give the vector an RMS magnitude of $\approx1$ before applying learned map $\gamma$
- Input is a hidden state tensor $X \in \mathbb{R}^{B\times C \times \mathtt{d\_model}}$ and the goal is to normalize each token representation $x_{b,j} \in \mathbb{R}^{\mathtt{d\_model}}$
- Compute RMS - root mean square magnitude for the token representation $$\text{RMS}(x_{b,j}) = \sqrt{\frac{1}{\mathtt{d\_model}}\sum_{r=1}^{\mathtt{d\_model}}x_{b,j}(r)^2}$$
- Use a learned coordinate vector $\gamma\in \mathbb{R}^{\mathtt{d\_model}}$ to scale each coordinate
- Compute the full normalized vector $\text{RMSNorm}(x_{b,j})\in \mathbb{R}^{\mathtt{d\_model}}$ for $r \in \{1,\dots, \mathtt{d\_model}\}$   
$$\text{RMSNorm}(x_{b,j})_r =\gamma(r) \cdot \frac{x_{b,j}(r)}{   \sqrt{\frac{1}{\mathtt{d\_model}}\sum_{k=1}^{\mathtt{d\_model}}x_{b,j}(k)^2 +\epsilon}}$$