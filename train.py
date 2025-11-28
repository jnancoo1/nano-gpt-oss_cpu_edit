from training.data_loader import train_loader, val_loader
from architecture.gptoss import Transformer, ModelConfig
import torch
from inference import generate_text

# Change device to CPU
device = "cpu"  # Changed from "cuda:0"

context = "Once upon a day"

# Create model on CPU
model = Transformer(
    ModelConfig(
        num_attention_heads=8,
        num_key_value_heads=4,
        num_experts=4,
        experts_per_token=1,
        num_hidden_layers=12,
        hidden_size=1024,
        intermediate_size=1024
    ),
    device
)

# Calculate parameters
print(sum([p.numel() for p in model.parameters()]) / 1000000, "M parameters")

# Save model
torch.save(model.state_dict(), "model/gptoss.pt")

# Generate text
generate_text(model, context)
