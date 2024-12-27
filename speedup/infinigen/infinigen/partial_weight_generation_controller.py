import torch
import torch.nn.functional as F


def partial_weight_index_generation(query, n_head, head_dim, partial_weight_ratio):
    """Generates the indices of partial weight query and partial key cache.

    On the prefill stage, generates the indices of partial weight query and
    partial key cache using the query matrix. By comparing the absolute sum of
    each column of the query matrix, gets the indices of top-k columns. These
    columns correspond to the columns that strongly affect the attention score.
    Thus, we use only those partial columns of query and key for speculation.

    Args:
        query: Query matrix (b, n, D)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d
        partial_weight_ratio: Ratio of the top-k columns

    Returns:
        partial_weight_index: Indices of top-k columns (b, h, d')
            where d' is d * (partial_weight_ratio).
    """

    partial_weight_index = torch.zeros(n_head, int(head_dim * partial_weight_ratio)).to(
        query.device
    )
    b = query.shape[0]

    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        sum_abs = torch.sum(torch.abs(query[0, :, start:end]), dim=-2)  # (Dh,)

        _, ind = torch.topk(
            sum_abs,
            int(head_dim * partial_weight_ratio),
        )

        partial_weight_index[h_idx] = ind

    return partial_weight_index.unsqueeze(0).repeat(b, 1, 1).to(torch.int64)


def set_partial_cache(k_cache, partial_index, n_head, head_dim):
    """Sets the partial key cache.

    On the prefill and decoding stages, generates the partial key cache
    following the partial_index which indicates the indices of the important
    columns.

    Args:
        k_cahce: Key cache (n, bh, d)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_cache: Partial key cache (n, bh, d')
    """

    n, bh, _ = k_cache.shape
    partial_cache = torch.gather(
        k_cache.view(n, -1, n_head, head_dim),
        3,
        partial_index.unsqueeze(0).repeat(n, 1, 1, 1),
    )
    return partial_cache.view(n, bh, -1)


def set_partial_weight(w_q, partial_index, n_head, head_dim):
    """Sets the partial query weight.

    On the prefill stage, generates the partial query weight following the
    partial_index which indicates the indices of the important columns.

    Args:
        w_q: Query weight (D, D)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_weight: Partial query weight (D', D)
    """
    partial_weight = F.embedding(
        partial_index[0]
        + torch.arange(n_head)[:, None].to(partial_index.device) * head_dim,
        w_q.view(-1, w_q.shape[-1]),
    )
    return partial_weight.view(-1, w_q.shape[-1])


# [x] For GQA
def partial_weight_index_generation_gqa(
    query, q_head_num, kv_head_num, head_dim, partial_weight_ratio
):
    """Generates the indices of partial weight query and partial key cache.

    On the prefill stage, generates the indices of partial weight query and
    partial key cache using the query matrix. By comparing the absolute sum of
    each column of the query matrix, gets the indices of top-k columns. These
    columns correspond to the columns that strongly affect the attention score.
    Thus, we use only those partial columns of query and key for speculation.

    ===GQA===
    The output partial indices are used in both Q and KV.
    But the function for MHA only considers Q.
    We need to modify the function to consider a group of queries.


    Args:
        query: Query matrix (b, n, D) => (b, n, Dh * h_q)
        q_head_num: Number of Q heads which we refer to as h_q
        kv_head_num: Number of KV heads which we refer to as h_kv
        head_dim: Hidden dimension of each head which we refer to as d
        partial_weight_ratio: Ratio of the top-k columns

    Returns:
        kv_partial_weight_index: Indices of top-k columns (b, h_kv, d')
        q_partial_weight_index: Indices of top-k columns (b, h_q, d')
            where d' is d * (partial_weight_ratio).
    """
    q_per_kv = q_head_num / kv_head_num

    kv_partial_weight_index = torch.zeros(
        kv_head_num, int(head_dim * partial_weight_ratio)
    ).to(query.device)
    q_partial_weight_index = torch.zeros(
        q_head_num, int(head_dim * partial_weight_ratio)
    ).to(query.device)
    b = query.shape[0]  # batch_size

    # Using mean for GQA
    for kv_h_idx in range(kv_head_num):
        q_sums = torch.zeros(q_per_kv, int(head_dim * partial_weight_ratio))
        q_base = kv_h_idx * q_per_kv

        for i in range(q_per_kv):
            start = (q_base + i) * head_dim
            end = (q_base + i + 1) * head_dim
            q_sums[i] = torch.sum(torch.abs(query[0, :, start:end]), dim=-2)
        # [ ] Design choice: Mean vs Max
        q_mean = torch.mean(q_sums, dim=0, keepdim=True)  # (Dh,)

        _, ind = torch.topk(
            q_mean,
            int(head_dim * partial_weight_ratio),
        )  # (topk_num,)
        kv_partial_weight_index[kv_h_idx] = ind

        for i in range(q_per_kv):
            q_h_idx = q_base + i
            q_partial_weight_index[q_h_idx] = ind

    return kv_partial_weight_index.unsqueeze(0).repeat(b, 1, 1).to(
        torch.int64
    ), q_partial_weight_index.unsqueeze(0).repeat(b, 1, 1).to(torch.int64)


# [x] For GQA
def set_partial_cache_gqa(k_cache, partial_index, kv_head_num, head_dim):
    """Sets the partial key cache.

    On the prefill and decoding stages, generates the partial key cache
    following the partial_index which indicates the indices of the important
    columns.

    Args:
        k_cache: Key cache (n, bh_kv, d)
        partial_weight_index: Indices of top-k columns (b, h_kv, d')
        kv_head_num: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d


    Returns:
        partial_cache: Partial key cache (n, bh_kv, d')
        where d' is d * (partial_weight_ratio).
    """

    n, bh_kv, _ = k_cache.shape
    partial_cache = torch.gather(
        k_cache.view(n, -1, kv_head_num, head_dim),  # (n, b, h_kv, d)
        3,
        partial_index.unsqueeze(0).repeat(n, 1, 1, 1),  # (n, d', 1, 1)
    )
    return partial_cache.view(n, bh_kv, -1)


# [x] For GQA
def set_partial_weight_gqa(w_q, partial_index, q_head_num, head_dim):
    """Sets the partial query weight.

    On the prefill stage, generates the partial query weight following the
    partial_index which indicates the indices of the important columns.

    Args:
        w_q: Query weight (D, D+1) where D is h_q * d
        partial_weight_index: Indices of top-k columns (b, h_q, d')
        q_head_num: Number of heads which we refer to as h_q
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_weight: Partial query weight (D', D+1)
    """

    partial_weight = F.embedding(
        partial_index[0]
        + torch.arange(q_head_num)[:, None].to(partial_index.device) * head_dim,
        w_q.view(-1, w_q.shape[-1]),
    )
    return partial_weight.view(-1, w_q.shape[-1])
