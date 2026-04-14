#!/usr/bin/env python3
"""Debug script to compare encoder outputs between Python and Rust implementations."""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, DebertaV2Model
from gliner import GLiNER

# Load the model
model = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
tokenizer = model.data_processor.transformer_tokenizer

# Text and labels
text = "Apple was founded by Steve Jobs in California."
entity_labels = ["person", "organization", "location"]
relation_labels = ["founded by", "located in"]

# Build the same prompt as our Rust code
# IMPORTANT: Don't add [CLS] - tokenizer adds it automatically
def build_prompt():
    """Build the prompt in the same format as Rust (without [CLS])."""
    parts = []
    for label in entity_labels:
        parts.append("<<ENT>>")
        parts.append(label)
    parts.append("[SEP]")
    for label in relation_labels:
        parts.append("<<REL>>")
        parts.append(label)
    parts.append("[SEP]")
    parts.append(text)
    return " ".join(parts)

prompt = build_prompt()
print(f"Prompt: {prompt}")

# Tokenize (tokenizer should add [CLS] automatically)
encoding = tokenizer(
    prompt,
    return_tensors="pt",
    padding=False,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

print(f"\nToken IDs (first 30): {input_ids[0, :30].tolist()}")
print(f"Token count: {input_ids.shape[1]}")

# Decode tokens to see what they are
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
print(f"\nFirst 30 tokens: {tokens[:30]}")

# Find <<ENT>> token positions
ent_token_id = tokenizer.convert_tokens_to_ids("<<ENT>>")
print(f"\n<<ENT>> token ID: {ent_token_id}")

ent_positions = []
for i, tok in enumerate(input_ids[0]):
    if tok.item() == ent_token_id:
        ent_positions.append(i)
print(f"<<ENT>> positions: {ent_positions}")

# Find the encoder - try different attribute names
print(f"\nModel type: {type(model)}")
print(f"Model attributes: {[a for a in dir(model) if not a.startswith('_')]}")

# The encoder is typically model.model in GLiNER
encoder = None
if hasattr(model, 'model'):
    encoder = model.model
    print(f"Found encoder at model.model: {type(encoder)}")
elif hasattr(model, 'token_rep_layer'):
    encoder = model.token_rep_layer
    print(f"Found encoder at model.token_rep_layer: {type(encoder)}")

# Check what encoder contains
if encoder is not None:
    print(f"Encoder attributes: {[a for a in dir(encoder) if not a.startswith('_')]}")

    # Try to find the actual DeBERTa model
    if hasattr(encoder, 'deberta'):
        deberta = encoder.deberta
        print(f"Found DeBERTa at encoder.deberta: {type(deberta)}")
    elif hasattr(encoder, 'model'):
        deberta = encoder.model
        print(f"Found DeBERTa at encoder.model: {type(deberta)}")
    else:
        deberta = encoder

# Run the encoder
with torch.no_grad():
    # Get the token_rep_layer (DeBERTa encoder)
    token_rep_layer = model.model.token_rep_layer
    print(f"\ntoken_rep_layer type: {type(token_rep_layer)}")
    print(f"token_rep_layer children: {list(token_rep_layer.named_children())}")

    # Call the token_rep_layer
    outputs = token_rep_layer(input_ids, attention_mask=attention_mask)
    if hasattr(outputs, 'last_hidden_state'):
        hidden_states = outputs.last_hidden_state
    elif isinstance(outputs, tuple):
        hidden_states = outputs[0]
    else:
        hidden_states = outputs

print(f"\nEncoder (token_rep_layer) output shape: {hidden_states.shape}")

# Stats
hs = hidden_states[0].numpy()
print(f"Encoder output stats: mean={hs.mean():.6f}, std={hs.std():.6f}, min={hs.min():.6f}, max={hs.max():.6f}")

# Print values at <<ENT>> positions
print(f"\nEncoder output at <<ENT>> positions (first 5 values):")
for pos in ent_positions:
    vals = hs[pos, :5]
    print(f"  pos {pos}: [{', '.join(f'{v:.4f}' for v in vals)}]")

# Also check other positions for comparison
print(f"\nEncoder output at other positions:")
for pos in [0, 2, 4, 10, 17]:
    if pos < hs.shape[0]:
        vals = hs[pos, :5]
        print(f"  pos {pos}: [{', '.join(f'{v:.4f}' for v in vals)}]")

# Check prompt_rep_layer
print("\n--- Prompt Rep Layer ---")
ent_embs = hidden_states[0, ent_positions, :]  # [n_labels, hidden]
print(f"Entity embeddings shape: {ent_embs.shape}")
print(f"Entity embeddings stats: mean={ent_embs.mean():.6f}, std={ent_embs.std():.6f}")

# Apply prompt_rep_layer from model.model
prompt_rep = model.model.prompt_rep_layer  # This should be a Sequential or MLP
projected = prompt_rep(ent_embs)
print(f"After prompt_rep_layer: shape={projected.shape}")
print(f"After prompt_rep_layer stats: mean={projected.mean():.6f}, std={projected.std():.6f}")

# Check what prompt_rep_layer consists of
print(f"\nPrompt rep layer structure:")
for name, module in prompt_rep.named_modules():
    if name:
        print(f"  {name}: {module}")

# Print projected values for first 5 dims
print(f"\nProjected entity embeddings (first 5 values):")
for i, label in enumerate(entity_labels):
    vals = projected[i, :5].detach().numpy()
    print(f"  {label}: [{', '.join(f'{v:.4f}' for v in vals)}]")

# Now let's check the raw token embeddings (before any attention)
print("\n--- Raw Token Embeddings (before transformer) ---")
bert_layer = token_rep_layer.bert_layer
deberta_model = bert_layer.model

# Get raw embeddings
word_embs = deberta_model.embeddings(input_ids)
print(f"Raw embeddings shape: {word_embs.shape}")

word_embs_np = word_embs[0].detach().numpy()
print(f"Raw embeddings stats: mean={word_embs_np.mean():.6f}, std={word_embs_np.std():.6f}")

print(f"\nRaw embeddings at <<ENT>> positions (first 5 values):")
for pos in ent_positions:
    vals = word_embs_np[pos, :5]
    print(f"  pos {pos}: [{', '.join(f'{v:.4f}' for v in vals)}]")

print(f"\nRaw embeddings at other positions (first 5 values):")
for pos in [0, 2, 4, 10, 17]:
    if pos < word_embs_np.shape[0]:
        vals = word_embs_np[pos, :5]
        print(f"  pos {pos}: [{', '.join(f'{v:.4f}' for v in vals)}]")
