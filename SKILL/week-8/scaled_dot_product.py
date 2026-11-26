
import torch
import math


def scaled_dot_product_attention(Q, K, V, mask=None):

    # Step 1: Calculate dot product of Q and K
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale the scores
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply Softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: Multiply weights with V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# Main program
if __name__ == "__main__":

    torch.manual_seed(42)

    Q = torch.rand(2, 3, 4)
    K = torch.rand(2, 3, 4)
    V = torch.rand(2, 3, 4)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("SCALED DOT PRODUCT ATTENTION")
    print("--------------------------------")

    print("Query Shape:", Q.shape)
    print("Key Shape:", K.shape)
    print("Value Shape:", V.shape)

    print("\nAttention Weights:")
    print(weights)

    print("\nAttention Output:")
    print(output)

    print("\nOutput Shape:", output.shape)
