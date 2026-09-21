# Method specification

All quantities below are computed independently at each valid response position.
Let $z$ be student logits, $q=\operatorname{softmax}(z)$, and $p$ the frozen teacher
distribution on the same token vocabulary. Logarithms are natural.

$$R=\sum_v q_v(\log q_v-\log p_v),\qquad
F=\sum_v p_v(\log p_v-\log q_v).$$

Their logit gradients are

$$g_R=q\odot(\log q-\log p-R),\qquad g_F=q-p.$$

Write $e_R=g_R/\|g_R\|_2$ and $e_F=g_F/\|g_F\|_2$, with zero gradients mapped
to zero directions by a safe denominator. Set $c=e_R^\top e_F$.

Entropy-informed priors use $H(r)=-\sum_v r_v\log r_v$ and
$s=H(q)/(H(q)+h_0)$, $t=H(p)/(H(p)+h_0)$:

$$a_R=\epsilon+(1-t)s,\qquad a_F=\epsilon+t,\qquad
\omega_R=\frac{a_R}{a_R+a_F},\quad\omega_F=\frac{a_F}{a_R+a_F}.$$

Defaults are $h_0=1$ and $\epsilon=0.05$. The Nash synthesis is

$$x=c(\omega_R-\omega_F),\qquad
k=\frac{x+\sqrt{x^2+4\omega_R\omega_F}}{2\omega_F},\qquad
d=\frac{k e_R+e_F}{\|k e_R+e_F\|_2}.$$

For negative $x$, the implementation uses the algebraically equivalent
$k=2\omega_R/(\sqrt{x^2+4\omega_R\omega_F}-x)$ to avoid cancellation.

The requested gate and final signal are

$$B=\frac12\sum_v(\sqrt{p_v}-\sqrt{q_v})^2,\qquad
w=\min(B/\tau,1),\qquad g=w\|g_F\|_2d.$$

The default is $\tau=0.01$, changed from the earlier experimental value of 0.05.
All three hyperparameters $h_0$, $\epsilon$ and $\tau$ must be finite and positive.

The implementation centers $g$ to remove floating-point residuals in its
zero-sum constraint. It uses a detached signal in the surrogate

$$L=\frac1N\sum_{i\in\mathcal M}\sum_v
\operatorname{stopgrad}(g_{i,v})\log q_{i,v}.$$

Since each signal sums to zero, this produces the intended student logit
gradient $g_i/N$. Gradient descent applies the negative of that gradient.
Identical teacher/student distributions produce a zero signal.

Float16/bfloat16 controller inputs are promoted to float32. If probabilities
underflow in float32, the computation retries in float64; underflow there is an
explicit error rather than an implicit support truncation. Non-finite logits
are rejected. No top-k approximation or gradient through controller coefficients
is part of this implementation.
