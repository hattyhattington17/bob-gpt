- Self attention is not aware of token positions, we must incorporate position information into each token representation
	- RoPE applies the position info to queries and keys after projection by forming pairs and rotating them
- Applied during self attention, after reshaping projections into heads $Q,K,V \in \mathbb{R}^{B\times H\times C\times \mathtt{d\textunderscore head}}$
	- RoPE is applied to each batch, context position $j$, and head $h$ to the key and query vectors
	- $q^h_j,k^h_j \in \mathbb{R}^{\mathtt{d\textunderscore head}}$
- Rotation - linear map that changes the direction of a vector without changing the length
	- Rotation in 2d by an angle $\theta$ of a pair $(a,b)$:

$$(a',b') = (a\cos{\theta} - b\sin{\theta}, a\sin{\theta} + b\cos{\theta})$$

- RoPE splits each vector into pairs of coordinates (2d vectors in $\mathbb{R}^{2}$) and rotates them, then concatenates them back into $\mathbb{R}^{\mathtt{d\textunderscore head}}$
	- $\mathtt{d\textunderscore head}$ needs to be even
# Steps
- Start with $q^h_j,k^h_j \in \mathbb{R}^{\mathtt{d\textunderscore head}}$
- Split the coordinates into $\mathtt{d\textunderscore head}/2$ pairs
	- $((q_0, q_1),(q_2, q_3),\dots, (q_{\mathtt{d\textunderscore head}-2}, q_{\mathtt{d\textunderscore head}-1}))$
	- $((k_0, k_1),(k_2, k_3),\dots, (k_{\mathtt{d\textunderscore head}-2}, k_{\mathtt{d\textunderscore head}-1}))$
- Compute an angle $\theta_{j,m}$ as a function of the position $j$ and pair index $m \in \{0,\dots, \frac{\mathtt{d\textunderscore head}}{2}-1\}$

$$\theta_{j,m} = j\omega_m$$

- $\omega_m$ is the frequency assigned to pair index $m$, $\mathtt{base}$ is the base constant in the frequency computation
	- often $\mathtt{base}=10000$

$$\omega_m = \mathtt{base}^{-\frac{2m}{\mathtt{d\textunderscore head}}}$$

- Rotate each pair by $\theta_{j,m}$
	- Pairs are $(q^{b,h,j}_{2m},q^{b,h,j}_{2m+1})$ and $(k^{b,h,j}_{2m},k^{b,h,j}_{2m+1})$
	- Rotated:
		- $(q^{b,h,j}_{2m}\cos{\theta_{j,m}} - q^{b,h,j}_{2m+1}\sin{\theta_{j,m}},\ q^{b,h,j}_{2m}\sin{\theta_{j,m}} + q^{b,h,j}_{2m+1}\cos{\theta_{j,m}})$
		- $(k^{b,h,j}_{2m}\cos{\theta_{j,m}} - k^{b,h,j}_{2m+1}\sin{\theta_{j,m}},\ k^{b,h,j}_{2m}\sin{\theta_{j,m}} + k^{b,h,j}_{2m+1}\cos{\theta_{j,m}})$
- Concatenate the rotated pairs back into $q^h_j,k^h_j \in \mathbb{R}^{\mathtt{d\textunderscore head}}$
# Cache
- The rotation angle $\theta_{j,m}$ is computed deterministically by position $j$ and pair index $m$, so we can cache a table of sin and cosine values and reuse them across batches and heads
- for all $j \in \{0,\dots, \mathtt{max\textunderscore seq\textunderscore len} -1 \}$, for all $m \in \{0,\dots, \frac{\mathtt{d\textunderscore head}}{2}-1\}$, compute

$$\theta_{j,m} = j\cdot \mathtt{base}^{-\frac{2m}{\mathtt{d\textunderscore head}}}$$

- store $\sin(\theta_{j,m})$ and $\cos(\theta_{j,m})$ in cache tables
	- $\mathtt{rope\textunderscore sin} \in \mathbb{R}^{\mathtt{max\textunderscore seq\textunderscore len} \times \frac{\mathtt{d\textunderscore head}}{2}}$
	- $\mathtt{rope\textunderscore cos} \in \mathbb{R}^{\mathtt{max\textunderscore seq\textunderscore len} \times \frac{\mathtt{d\textunderscore head}}{2}}$
