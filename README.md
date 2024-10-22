### Grouped-Query Attention (GQA) 
#### GQA in Transformers: Comparison with Multi-Head and Multi-Query Attention (Ainslie et al., 2023)

This repository explores Grouped-Query Attention (GQA), an efficient variation of the attention mechanism in transformers that balances the trade-off between memory usage and model performance. GQA generalizes both Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by sharing key-value pairs across groups of query heads.
Contents:

1. Jupyter Notebook\
   
The notebook provides an in-depth explanation of how GQA works, and compares it to MHA and MQA. It includes:
An overview of the challenges in scaling transformers, particularly with attention mechanisms.
Detailed comparisons between:
Multi-Head Attention (MHA): Each query head has its own key-value pair.
Multi-Query Attention (MQA): All query heads share a single key-value pair, reducing memory usage.
Grouped-Query Attention (GQA): Query heads are grouped, and each group shares a key-value pair, which provides a middle ground between MHA and MQA in terms of memory usage and performance.
Mathematical explanations and code snippets that show how the attention mechanisms differ.
Visualizations of the results and comparisons of the computational efficiency of GQA vs MHA and MQA.

3. Pseudocode
   
The repository also includes pseudocode for GQA, illustrating how the query, key, and value heads interact in this generalized attention mechanism. The pseudocode explains how:
Query heads are grouped.
Key and value heads are shared within groups.
The attention mechanism computes attention scores using shared key-value pairs.
You can find this pseudocode bellow.

5. Repository Files
   
- Grouped Querry Attention.ipynb: The Jupyter notebook with detailed explanations, comparisons, and visualizations.
- README.md: This file, providing a summary of the project and links to the notebook.
- Algorithms for transformers.pdf. An article explaining the architecture in transformers.
- [The original GQA paper by Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)

6. Additional Resources

- [Towards Data Science Overview](https://towardsdatascience.com/demystifying-gqa-grouped-query-attention-3fb97b678e4a)
- [Variants of Multi-Head Attention Video] (https://www.youtube.com/watch?v=pVP0bu8QA2w)
- [Runtime reproduction expirement](https://github.com/fkodom/grouped-query-attention-pytorch)
- [As seen in Lama2](https://github.com/meta-llama/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py)
#### Pseudocode of GQA
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
