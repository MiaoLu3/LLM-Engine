"""
Model loading from HuggingFace.

Loads Qwen3 (or other decoder-only transformers) and extracts
key model dimensions needed for KV cache allocation.

Also provides functions to load weights from HuggingFace checkpoints
into our custom model implementations.
"""

import glob
import json
from pathlib import Path
from typing import Tuple, Dict, TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from llm_engine.config import ModelConfig

if TYPE_CHECKING:
    from llm_engine.model.qwen3 import Qwen3ForCausalLM


class ModelLoader:
    """
    Loads a HuggingFace model and extracts architecture information.

    This is a thin wrapper that:
    1. Loads model weights via AutoModelForCausalLM
    2. Extracts key dimensions (num_layers, num_heads, head_dim, etc.)
    3. Moves model to specified device with specified dtype
    """

    @staticmethod
    def load(
        config: ModelConfig,
        device: str = "cuda",
    ) -> Tuple[torch.nn.Module, ModelConfig]:
        """
        Load model and populate ModelConfig with derived attributes.

        Args:
            config: Model configuration with model_name_or_path.
            device: Device to load model on.

        Returns:
            Tuple of (loaded model, updated config with derived attributes).
        """
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Load model configuration first to get architecture info
        hf_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Extract model dimensions from config
        # Different model families use different attribute names
        config.num_layers = _get_num_layers(hf_config)
        config.num_heads = _get_num_attention_heads(hf_config)
        config.num_kv_heads = _get_num_kv_heads(hf_config)
        config.hidden_size = hf_config.hidden_size
        config.head_dim = config.hidden_size // config.num_heads
        config.vocab_size = hf_config.vocab_size

        # Set max_model_len if not specified
        if config.max_model_len is None:
            config.max_model_len = _get_max_model_len(hf_config)

        # Load the actual model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            device_map=device,
        )

        # Set to eval mode
        model.eval()

        return model, config

    @staticmethod
    def get_memory_footprint(model: torch.nn.Module) -> int:
        """
        Calculate approximate GPU memory used by model weights.

        Args:
            model: Loaded PyTorch model.

        Returns:
            Memory usage in bytes.
        """
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes


def _get_num_layers(hf_config) -> int:
    """Extract number of transformer layers from HF config."""
    # Try different attribute names used by various models
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(hf_config, attr):
            return getattr(hf_config, attr)
    raise ValueError("Could not determine number of layers from model config")


def _get_num_attention_heads(hf_config) -> int:
    """Extract number of attention heads from HF config."""
    for attr in ["num_attention_heads", "n_head", "num_heads"]:
        if hasattr(hf_config, attr):
            return getattr(hf_config, attr)
    raise ValueError("Could not determine number of attention heads from model config")


def _get_num_kv_heads(hf_config) -> int:
    """
    Extract number of key-value heads from HF config.

    For models with Grouped Query Attention (GQA), this may differ
    from num_attention_heads. For MHA, they're equal.
    """
    # Qwen uses num_key_value_heads
    if hasattr(hf_config, "num_key_value_heads"):
        return hf_config.num_key_value_heads

    # Some models use n_head_kv
    if hasattr(hf_config, "n_head_kv"):
        return hf_config.n_head_kv

    # Fall back to num_attention_heads (MHA)
    return _get_num_attention_heads(hf_config)


def _get_max_model_len(hf_config) -> int:
    """Extract maximum context length from HF config."""
    # Try various attribute names
    for attr in [
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ]:
        if hasattr(hf_config, attr):
            return getattr(hf_config, attr)

    # Default fallback
    return 4096


def compute_kv_cache_size_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    """
    Compute KV cache memory per token in bytes.

    KV cache stores key and value tensors for each layer:
    - Keys: [num_kv_heads, head_dim] per token per layer
    - Values: [num_kv_heads, head_dim] per token per layer

    Args:
        config: Model configuration with architecture info.
        dtype: Data type for KV cache.

    Returns:
        Bytes per token for KV cache.
    """
    bytes_per_element = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
    }.get(dtype, 2)

    # Key + Value for each layer
    kv_size_per_layer = 2 * config.num_kv_heads * config.head_dim * bytes_per_element
    total_size = config.num_layers * kv_size_per_layer

    return total_size


def compute_available_kv_blocks(
    available_memory_bytes: int,
    config: ModelConfig,
    block_size: int,
    dtype: torch.dtype,
) -> int:
    """
    Compute number of KV cache blocks that fit in available memory.

    Args:
        available_memory_bytes: Available GPU memory for KV cache.
        config: Model configuration.
        block_size: Tokens per block.
        dtype: Data type for KV cache.

    Returns:
        Number of blocks that can be allocated.
    """
    bytes_per_token = compute_kv_cache_size_per_token(config, dtype)
    bytes_per_block = bytes_per_token * block_size
    return available_memory_bytes // bytes_per_block


def load_qwen3_weights(
    model: "Qwen3ForCausalLM",
    model_path: str,
) -> None:
    """
    Load weights from HuggingFace checkpoint into our custom Qwen3 model.

    Supports loading from:
    - safetensors files (preferred)
    - pytorch_model.bin files (fallback)

    The weight names in HuggingFace Qwen3 match our model structure:
    - model.embed_tokens.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.layers.{i}.mlp.{gate,up,down}_proj.weight
    - model.norm.weight
    - lm_head.weight

    Args:
        model: Our custom Qwen3ForCausalLM model instance.
        model_path: Path to model directory containing weight files.
    """
    model_path = Path(model_path)

    # Check for safetensors files first (preferred format)
    safetensor_files = list(model_path.glob("*.safetensors"))

    if safetensor_files:
        _load_from_safetensors(model, safetensor_files)
    else:
        # Fall back to pytorch_model.bin
        pytorch_files = list(model_path.glob("pytorch_model*.bin"))
        if pytorch_files:
            _load_from_pytorch_bin(model, pytorch_files)
        else:
            # Try model.safetensors or pytorch_model.bin as single files
            single_safetensor = model_path / "model.safetensors"
            single_pytorch = model_path / "pytorch_model.bin"

            if single_safetensor.exists():
                _load_from_safetensors(model, [single_safetensor])
            elif single_pytorch.exists():
                _load_from_pytorch_bin(model, [single_pytorch])
            else:
                raise FileNotFoundError(
                    f"No weight files found in {model_path}. "
                    "Expected *.safetensors or pytorch_model*.bin"
                )


def _load_from_safetensors(
    model: "Qwen3ForCausalLM",
    safetensor_files: list,
) -> None:
    """
    Load weights from safetensors files.

    Args:
        model: Our custom Qwen3ForCausalLM model instance.
        safetensor_files: List of paths to safetensor files.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "safetensors is required for loading .safetensors files. "
            "Install with: pip install safetensors"
        )

    # Build mapping from HF names to our model parameter names
    weight_map = _build_qwen3_weight_map(model)

    # Track which weights we've loaded
    loaded_weights = set()
    model_state_dict = model.state_dict()

    # Load from each safetensor file
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                if hf_name in weight_map:
                    our_name = weight_map[hf_name]
                    tensor = f.get_tensor(hf_name)

                    # Verify shape matches
                    if our_name in model_state_dict:
                        expected_shape = model_state_dict[our_name].shape
                        if tensor.shape != expected_shape:
                            raise ValueError(
                                f"Shape mismatch for {hf_name} -> {our_name}: "
                                f"got {tensor.shape}, expected {expected_shape}"
                            )

                    # Load the weight
                    _set_weight(model, our_name, tensor)
                    loaded_weights.add(our_name)

    # Check for missing weights
    missing_weights = set(model_state_dict.keys()) - loaded_weights
    if missing_weights:
        # Filter out expected missing weights (e.g., tied embeddings)
        critical_missing = [
            w for w in missing_weights
            if not _is_expected_missing(w, model)
        ]
        if critical_missing:
            print(f"Warning: {len(critical_missing)} weights not loaded: "
                  f"{critical_missing[:5]}...")


def _load_from_pytorch_bin(
    model: "Qwen3ForCausalLM",
    pytorch_files: list,
) -> None:
    """
    Load weights from pytorch_model.bin files.

    Args:
        model: Our custom Qwen3ForCausalLM model instance.
        pytorch_files: List of paths to pytorch bin files.
    """
    weight_map = _build_qwen3_weight_map(model)
    loaded_weights = set()
    model_state_dict = model.state_dict()

    for pt_path in pytorch_files:
        state_dict = torch.load(pt_path, map_location="cpu")

        for hf_name, tensor in state_dict.items():
            if hf_name in weight_map:
                our_name = weight_map[hf_name]

                if our_name in model_state_dict:
                    expected_shape = model_state_dict[our_name].shape
                    if tensor.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {hf_name} -> {our_name}: "
                            f"got {tensor.shape}, expected {expected_shape}"
                        )

                _set_weight(model, our_name, tensor)
                loaded_weights.add(our_name)

    missing_weights = set(model_state_dict.keys()) - loaded_weights
    if missing_weights:
        critical_missing = [
            w for w in missing_weights
            if not _is_expected_missing(w, model)
        ]
        if critical_missing:
            print(f"Warning: {len(critical_missing)} weights not loaded: "
                  f"{critical_missing[:5]}...")


def _build_qwen3_weight_map(model: "Qwen3ForCausalLM") -> Dict[str, str]:
    """
    Build mapping from HuggingFace weight names to our model parameter names.

    HuggingFace Qwen3 uses the same naming convention as our model,
    so this is mostly an identity mapping with some minor adjustments.

    Returns:
        Dictionary mapping HF name -> our model name.
    """
    weight_map = {}
    num_layers = model.config.num_hidden_layers

    # Embeddings
    weight_map["model.embed_tokens.weight"] = "model.embed_tokens.weight"

    # Layers
    for i in range(num_layers):
        prefix_hf = f"model.layers.{i}"
        prefix_ours = f"model.layers.{i}"

        # Layer norms
        weight_map[f"{prefix_hf}.input_layernorm.weight"] = \
            f"{prefix_ours}.input_layernorm.weight"
        weight_map[f"{prefix_hf}.post_attention_layernorm.weight"] = \
            f"{prefix_ours}.post_attention_layernorm.weight"

        # Attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_map[f"{prefix_hf}.self_attn.{proj}.weight"] = \
                f"{prefix_ours}.self_attn.{proj}.weight"

        # Q/K norms (if present in HF model)
        weight_map[f"{prefix_hf}.self_attn.q_norm.weight"] = \
            f"{prefix_ours}.self_attn.q_norm.weight"
        weight_map[f"{prefix_hf}.self_attn.k_norm.weight"] = \
            f"{prefix_ours}.self_attn.k_norm.weight"

        # MLP
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            weight_map[f"{prefix_hf}.mlp.{proj}.weight"] = \
                f"{prefix_ours}.mlp.{proj}.weight"

    # Final norm and lm_head
    weight_map["model.norm.weight"] = "model.norm.weight"
    weight_map["lm_head.weight"] = "lm_head.weight"

    return weight_map


def _set_weight(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """
    Set a weight in the model by its full dotted name.

    Args:
        model: The model to set the weight in.
        name: Full dotted name (e.g., "model.layers.0.self_attn.q_proj.weight").
        tensor: The tensor value to set.
    """
    parts = name.split(".")
    obj = model

    # Navigate to the parent module
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    # Set the parameter
    param_name = parts[-1]
    param = getattr(obj, param_name)

    if isinstance(param, torch.nn.Parameter):
        param.data.copy_(tensor)
    else:
        setattr(obj, param_name, tensor)


def _is_expected_missing(weight_name: str, model: "Qwen3ForCausalLM") -> bool:
    """
    Check if a weight is expected to be missing (e.g., tied weights).

    Args:
        weight_name: Name of the weight.
        model: The model.

    Returns:
        True if the weight is expected to be missing.
    """
    # lm_head may be tied to embed_tokens
    if weight_name == "lm_head.weight" and model.config.tie_word_embeddings:
        return True

    # Q/K norms may not be present in all Qwen3 variants
    if "q_norm" in weight_name or "k_norm" in weight_name:
        return True

    return False
