### Grouped-Query Attention (GQA) Pseudocode
```python
# Parameters:
# Q_heads: List of query heads (size H)
# K_heads: List of key heads (size H)
# V_heads: List of value heads (size H)
# G: Number of groups (1 <= G <= H)
# Step 1: Group Query Heads and Mean-Pool Key-Value Heads
group_size = H // G  # Size of each group (assume H divisible by G)
grouped_K = []  # Initialize list to store grouped keys
grouped_V = []  # Initialize list to store grouped values
for i in range(G):
    # Get the query heads for this group
    query_group = Q_heads[i * group_size : (i + 1) * group_size]
    # Mean-pool the key and value heads for the group
    group_key = mean_pool(K_heads[i * group_size : (i + 1) * group_size])
    group_value = mean_pool(V_heads[i * group_size : (i + 1) * group_size])
    grouped_K.append(group_key)  # Store the pooled key for this group
    grouped_V.append(group_value)  # Store the pooled value for this group
# Step 2: Compute Attention for Each Query Head
output_heads = []  # Initialize list to store output of each head
for i in range(H):
    # Determine which group the query head belongs to
    group_index = i // group_size
    # Get the corresponding shared key and value for the group
    key = grouped_K[group_index]
    value = grouped_V[group_index]
    # Compute attention score (dot product of query and key)
    attention_score = dot_product(Q_heads[i], key)
    # Compute the weighted sum of values based on the attention score
    output = attention_score * value
    # Store the output for this query head
    output_heads.append(output)
# Step 3: Concatenate the outputs of all heads
final_output = concatenate(output_heads)
# Return the final result of grouped-query attention
return final_output
