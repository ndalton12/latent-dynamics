from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class ModelSpec:
    path: str
    dtype: str | None = None
    load_model_kwargs: dict[str, Any] | None = None
    load_tokenizer_kwargs: dict[str, Any] | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gemma3_270m": ModelSpec(
        path="google/gemma-3-270m-it",
        dtype="bfloat16",
    ),
    "gemma3_4b": ModelSpec(
        path="google/gemma-3-4b-it",
        dtype="bfloat16",
    ),
    "llama3.2_1b": ModelSpec(
        path="meta-llama/Llama-3.2-1B-Instruct",
        dtype="bfloat16",
    ),
    "llama3.1_8b": ModelSpec(
        path="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
    ),
    "qwen3_0.6b": ModelSpec(
        path="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
    ),
    "qwen3_8b": ModelSpec(
        path="Qwen/Qwen3-8B",
        dtype="bfloat16",
    ),
}


def get_torch_dtype(dtype: torch.dtype | str | None) -> torch.dtype | str | None:
    """Convert a string or torch.dtype to a torch.dtype."""
    if dtype is None or dtype == "auto":
        return dtype

    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype: torch.{dtype}")

    return dtype


def load_model_and_tokenizer(
    model_spec: ModelSpec | str,
    device: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer from a ModelSpec."""
    if isinstance(model_spec, str):
        model_spec = MODEL_REGISTRY[model_spec]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_spec.path, **(model_spec.load_tokenizer_kwargs or {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.path,
        dtype=get_torch_dtype(model_spec.dtype),
        low_cpu_mem_usage=True,
        **(model_spec.load_model_kwargs or {}),
    )
    model.to(device)
    model.eval()

    print(
        f"Loaded model: {model_spec.path}"
        f"\n  Number of hidden layers:         {model.config.num_hidden_layers}"
        f"\n  Size of hidden layers:           {model.config.hidden_size}"
        f"\n  Size of activations (per token): {(model.config.num_hidden_layers + 1) * model.config.hidden_size * 4 // 1024} KB"
        f"\n  Model dtype:                     {model.dtype}"
        f"\n  Device:                          {model.device}"
    )
    return model, tokenizer


def detect_token_groups(tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    tokens = np.array(tokenizer.convert_ids_to_tokens(range(len(tokenizer))))
    tokens_spaces = set(
        [tokenizer.convert_ids_to_tokens(tokenizer.encode(t, add_special_tokens=False))[0] for t in [" ", "\n", "\t"]]
    )
    indices_special = tokenizer.all_special_ids
    indices_added = list(tokenizer.get_added_vocab().values())
    indices_spaces = [i for i, t in enumerate(tokens) if set(t) <= tokens_spaces]
    indices_digits = [i for i, t in enumerate(tokens) if t.isdigit() or (t.startswith("-") and t[1:].isdigit())]
    return {
        "special": indices_special,
        "added": indices_added,
        "digits": indices_digits,
        "spaces": indices_spaces,
    }


def get_token_groups(tokenizer: PreTrainedTokenizerBase) -> dict[str, list[int]]:
    tokenizer_id = f"{tokenizer.__class__.__name__}_{tokenizer.vocab_size}"
    if tokenizer_id == "GemmaTokenizer_262144":
        token_groups = {
            "common": [],  # set automatically
            "unused": list(range(6, 105)) + list(range(256001, 262144)),
            "spaces": list(range(107, 168)) + [236743] + list(range(255968, 255999)),
            "digits": [19411, 20331, 34728, 34930, 35930, 38743, 43947, 47714, 55727, 56886, 57585, 59814, 62505, 64406, 73733, 74408, 77441, 78308, 78760, 85781, 87834, 88716, 92624, 93924, 99250, 100951, 104365, 109151, 114486, 122920, 122983, 132722, 137728, 138839, 139493, 143992, 145709, 152268, 164070, 165315, 173374, 175246, 175639, 176250, 176452, 182156, 183527, 189949, 194556, 195081, 196135, 197192, 200085, 200170, 201324, 208476, 213522, 213960, 214147, 220858, 221619, 222694, 223403, 225716, 226128, 227090, 227342, 234151, 234725] + list(range(236770, 236772)) + [236778, 236800, 236810, 236812, 236819, 236825, 236828, 236832, 237269, 237361, 237425, 237749, 237827, 237832, 237837, 237846, 238009, 238016, 238131, 238204, 238254, 238456, 238522, 238540, 238647, 238658, 238684, 238750, 238787, 238964, 238995, 239146, 239158, 239311, 239325, 239341, 239374, 239414, 239416, 239503, 239580, 239689, 239784, 239836, 240010, 240038, 240161, 240197, 240290, 240302, 240366, 240394, 240535, 240567, 240595, 240929, 240949, 241129, 241303, 241388, 241415, 241685, 241848, 241928, 242044, 242086, 242090, 242097, 242172, 242205, 242300, 242323, 242526, 242553, 242758, 242849] + list(range(242862, 242864)) + [242899, 243123, 243319, 243397, 243471, 243489, 243542, 243586, 243609, 243639, 243687, 243791, 243926, 243971, 244005, 244044, 244099, 244139, 244326, 244335, 244337, 244345, 244347, 244443, 244467, 244551, 244705, 244816, 244976, 245019, 245051, 245224, 245327, 245469, 245532, 245771, 245800, 245811, 245823, 245832, 245835, 245930, 245959, 246117, 246285, 246320, 246490, 246495, 246579, 246784, 246869, 246878, 246976, 246983, 247101, 247147, 247214, 247233, 247276, 247353, 247462, 247822] + list(range(247840, 247842)) + [247855, 247909, 247918, 247964, 248101, 248118, 248283, 248429, 248460, 248691, 248740, 248829, 248866, 248955, 248994, 249046, 249058, 249065, 249098, 249132, 249165, 249412, 249516, 249518, 249528] + list(range(249593, 249595)) + [249751, 249780, 249892, 249999] + list(range(250047, 250049)) + [250122, 250272, 250302, 250312, 250320, 250341, 250729, 250755, 250852, 250857, 250937, 250973, 251124, 251164, 251299, 251320, 251455, 251460, 251492, 251661, 251700, 252024, 252053, 252327, 252371, 252381, 252497, 252557, 252573, 252601, 252752, 252828, 253304, 253417, 253482] + list(range(253654, 253656)) + [253859, 253881, 254091, 254297, 254618, 254654, 254762] + list(range(254847, 254849)) + [254923, 254932, 254974, 255015, 255063, 255189, 255272, 255276, 255389, 255472, 255839, 255854, 255879],
            "special": list(range(0, 6)) + [105, 106] + [255999, 256000],
            "bytes": list(range(238, 494)),
            "html": list(range(168, 238)),
        }  # fmt: skip
    elif tokenizer_id == "Qwen2Tokenizer_151643":
        token_groups = {
            "common": [],  # set automatically
            "unused": [],
            "spaces": list(range(197, 199)) + [220] + list(range(256, 258)) + [260, 262, 271, 286, 298, 310, 338, 394, 414, 464, 503, 571, 664, 688, 715, 786, 981, 999, 1022, 1060, 1066, 1144, 1383, 1406, 1572, 1662, 1698, 1789, 1797, 1843, 1920, 2187, 2290, 2303, 2394, 2549, 2559, 2683, 2760, 3344, 3374, 3502, 3677, 3824, 4293, 4557, 4569, 4597, 4710, 5108, 5134, 5180, 5238, 5401, 5872, 5959, 5968, 6360, 6374, 6449, 6526, 6656, 6926, 7018, 7213, 7451, 7472, 7561, 7631, 7782, 7847, 8136, 8333, 8689, 8945, 9103, 9359, 9401, 9699, 10179, 10503, 10589, 10683, 10947, 11070, 11120, 11787, 11869, 11950, 12306, 12573, 12841, 13043, 13063, 13463, 13544, 13693, 13887, 14265, 14621, 14642, 14731, 14808, 15270, 15287, 15429, 15677, 15799, 15865, 16159, 16693, 16885, 17264, 17362, 17476, 17546, 17642, 17648, 18007, 18236, 18325, 18363, 18445, 18574, 18611, 18749, 19011, 19205, 19271, 19273, 20295, 20835, 20974, 21497, 21509, 22207, 22335, 22701, 23263, 23292, 23419, 23459, 24178, 24348, 24520, 24616, 25435, 25773, 26065, 26285, 26546, 26723, 26809, 26921, 27352, 27668, 28247, 28666, 28802, 29936, 30016, 30363, 30417, 30645, 30711, 30779, 30930, 31044, 31670, 31906, 31979, 32678, 32717, 32814, 33641, 33933, 34135, 34149, 34483, 34513, 34583, 34642, 35117, 35329, 35557, 35627, 36521, 36577, 36624, 36845, 36920, 37083, 37144, 37204, 37692, 38171, 38320, 38458, 38484, 38812, 39484, 39767, 39865, 40104, 40337, 41056, 41636, 41693, 41982, 42708, 43211, 45128, 45807, 46145, 46256, 46452, 46464, 46600, 46771, 47123, 47549, 47930, 48180, 48426, 48892, 49106, 49987, 50233, 50440, 50538, 51068, 51093, 51124, 51141, 51370, 51475, 51480, 52078, 52720, 54060, 54712, 54833, 55447, 55799, 56546, 56596, 56940, 57041, 57654, 58087, 58247, 58591, 58667, 58958, 59101, 59649, 59949, 60123, 60998, 61439, 62372, 63477, 64204, 64568, 65226, 65267, 65271] + list(range(65496, 65498)) + [65668, 66376, 66526, 66828, 66834, 67392, 67664, 68203, 68546, 68973, 69562, 69877, 70057, 70577, 71248, 71400, 71664, 71779, 73196, 73363, 73427, 73531, 73912, 74376, 74525, 74568, 74904, 75228, 75305, 76325, 76366, 76639, 77787, 77993, 78099, 78672, 78702, 79000, 79039, 79083, 79133, 79226, 79524, 79682, 79871, 80445, 80719, 80755, 80840, 81048, 81221, 82361, 83268, 83795, 83913, 85065, 85422, 86359, 86766, 86770, 86827, 87079, 87728, 88044, 88804, 88866, 88901, 88998, 89253, 90075, 90306, 90358, 91151, 91214, 91693, 91737, 92163, 93004, 94779, 94947, 95087, 95429, 95805, 96017, 96090, 96094, 96678, 96800, 96991, 97417, 98207, 98251, 98738, 98878],
            "digits": list(range(15, 25)) + list(range(110, 112)) + [117],
            "special": list(range(151643, 151669)),
        }  # fmt: skip
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer_id}. Please define token groups for this tokenizer.")

    token_ids_excluded = set(sum(token_groups.values(), []))
    token_groups["common"] = [i for i in range(tokenizer.vocab_size) if i not in token_ids_excluded]
    return token_groups


def print_token_groups(
    tokenizer: PreTrainedTokenizerBase,
    token_groups: dict[str, list[int]],
    mode: Literal["dataframe", "array"] = "dataframe",
):
    for group_name, token_ids in token_groups.items():
        print(f"Token group: {group_name} ({len(token_ids)} tokens)")
        if mode == "dataframe":
            display(pd.DataFrame({"token_id": token_ids, "token": tokenizer.convert_ids_to_tokens(token_ids)}))
        elif mode == "array":
            print(np.array(tokenizer.convert_ids_to_tokens(token_ids)))
        elif mode == "ids":
            # Group consecutive IDs into sub-lists
            token_ids.sort()
            groups = []
            for x in token_ids:
                if groups and x == groups[-1][-1] + 1:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            # Format groups as ranges or lists of singletons
            segments = []
            for g in groups:
                if len(g) > 1:
                    segments.append((g[0], g[-1] + 1))
                elif segments and isinstance(segments[-1], list):
                    segments[-1].append(g[0])
                else:
                    segments.append(g)
            # Print the final joined string
            print(" + ".join(f"list(range{s})" if isinstance(s, tuple) else str(s) for s in segments))
        else:
            raise ValueError(f"Unknown mode {mode}. Supported modes: dataframe, array.")


def get_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int] | None = None,
) -> tuple[list[str], np.array]:
    if token_ids is None:
        token_ids = list(range(tokenizer.vocab_size))

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    embeddings = model.get_input_embeddings().weight.cpu().float().numpy()[token_ids]

    if model.__class__.__name__.startswith("Gemma3"):
        # Gemma3 models scale token embeddings by sqrt(hidden_size)
        # reference: https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/gemma3/modeling_gemma3.py#L102
        embeddings *= np.sqrt(model.config.hidden_size)

    print(
        f"Loaded token embeddings: {model.name_or_path}"
        f"\n  Number of tokens: {len(tokens)}"
        f"\n  Shape of embeddings: {embeddings.shape[1]}"
        f"\n  Size of embeddings: {embeddings.nbytes / 1024**2:.2f} MB"
    )
    return tokens, embeddings
