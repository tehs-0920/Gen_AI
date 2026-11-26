
import torch
import torch.nn as nn

from scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Q, K and V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Final output layer
        self.output = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):

        batch_size, seq_length, embed_dim = x.shape

        x = x.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x, mask=None):

        # Generate Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply scaled dot product attention
        attention_output, attention_weights = (
            scaled_dot_product_attention(Q, K, V, mask)
        )

        # Combine attention heads
        attention_output = attention_output.transpose(1, 2)

        batch_size, seq_length, _, _ = attention_output.shape

        attention_output = attention_output.contiguous().reshape(
            batch_size,
            seq_length,
            self.embed_dim
        )

        # Final linear projection
        output = self.output(attention_output)

        return output, attention_weights


# Main program
if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2
    seq_length = 5
    embed_dim = 8
    num_heads = 2

    x = torch.rand(batch_size, seq_length, embed_dim)

    model = MultiHeadAttention(embed_dim, num_heads)

    output, weights = model(x)

    print("MULTI-HEAD ATTENTION")
    print("--------------------------------")

    print("Input Shape:", x.shape)
    print("Number of Heads:", num_heads)
    print("Embedding Dimension:", embed_dim)

    print("\nAttention Weights Shape:", weights.shape)

    print("\nAttention Output:")
    print(output)

    print("\nOutput Shape:", output.shape)
