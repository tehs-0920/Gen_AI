
import torch

from scaled_dot_product import scaled_dot_product_attention
from multi_head_attention import MultiHeadAttention


def main():

    torch.manual_seed(42)

    print("======================================")
    print(" ATTENTION MECHANISM IMPLEMENTATION")
    print("======================================")

    # 1. Scaled Dot-Product Attention

    print("\n1. SCALED DOT PRODUCT ATTENTION")
    print("--------------------------------------")

    Q = torch.rand(2, 3, 4)
    K = torch.rand(2, 3, 4)
    V = torch.rand(2, 3, 4)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Query Shape:", Q.shape)
    print("Key Shape:", K.shape)
    print("Value Shape:", V.shape)

    print("\nAttention Weights:")
    print(weights)

    print("\nOutput Shape:", output.shape)

    # 2. Multi-Head Attention

    print("\n2. MULTI-HEAD ATTENTION")
    print("--------------------------------------")

    x = torch.rand(2, 5, 8)

    model = MultiHeadAttention(
        embed_dim=8,
        num_heads=2
    )

    output, weights = model(x)

    print("Input Shape:", x.shape)
    print("Number of Heads: 2")

    print("\nAttention Weights Shape:", weights.shape)

    print("\nMulti-Head Attention Output:")
    print(output)

    print("\nOutput Shape:", output.shape)

    print("\n======================================")
    print(" IMPLEMENTATION COMPLETED SUCCESSFULLY")
    print("======================================")


if __name__ == "__main__":
    main()
