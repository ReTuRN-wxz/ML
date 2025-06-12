# GNN

As we all know, this topic focuses on machine learning with graphs. The first question is: **How can a machine accept a graph as input?** 

Given that memory is expensive, we likely cannot use adjacent matrix which requires $O(|V|^2)$ storeage space, we need to consider more efficient ways to represent the graph. One initial idea is to assign each node a fixed-size vector.

But how? We must preserve important graph features, such as connectivity, during this process. The first attempt could involve explicitly maintaining edge information, ensuring that the encoded graph can be decoded while retaining most of its original structure, including edges.

### Random-Walk Embedding

Like any ML task, we need to design a loss function.

$$
\mathcal{L}=\sum_{u \in V}\sum_{v \in N_{R}(u)}-\log(P(v \mid \mathbf{z}_u))
$$

We should first explain why, our aim is to predict $u$ 's neighbour from $\mathbf{z}_u$ , so we want $P(v \mid \mathbf{z}_u)$ to be large, and in ML, our aim is to minimize the loss function, then we add $-$ to solve this problem.

The second step, what's $P(v \mid \mathbf{z}_u)$ ?

$$
P(v \mid \mathbf{z}_u)=\frac{\exp (\mathbf{z}_v^{T} \mathbf{z}_u)}{\sum_{w \in V} \exp(\mathbf{z}_{w}^{T}\mathbf{z}_u)}
$$

However, computing this probability naively is prohibitively expensive:

Caluculate the dot product costs $O(|V|)$ , and all nodes lead to $O(|V|^2)$ , we have to find a easier way to calculate it.

The logic behind the initial equation is that we first pick the positive sample, then we take some negative samples to evaluate it, so now we can reformulate $P$ into:

$$
\log \sigma(\mathbf{z}_v^{T} \mathbf{z}_u)+\sum_{k=1}^{K} \log \sigma(-z_{w_k}^{T}z_u)
$$

where $\sigma$ is the sigmoid function, now the loss function changes into:
$$
\mathcal{L}=-\sum_{u \in V} \sum_{w \in N_{R}(u)}\left[\log \sigma(\mathbf{z}_v^{T} \mathbf{z}_u)+\sum_{k=1}^{K} \log \sigma(-z_{w_k}^{T}z_u) \right]
$$
now it only needs $O(K|V|)$ to calculate the loss function.
