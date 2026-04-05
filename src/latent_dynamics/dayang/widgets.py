import contextlib
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from natsort import natsort_keygen
from traitlets import Bool

from latent_dynamics.dayang.activations import Activations, PoolMethod, extract_activations
from latent_dynamics.dayang.data import DATASET_REGISTRY, load_dataset_from_spec
from latent_dynamics.dayang.model import MODEL_REGISTRY, get_token_embeddings, load_model_and_tokenizer
from latent_dynamics.dayang.utils import select


def _str_to_slice(s: str) -> slice | None:
    return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(":")))


def get_activation_folders(path: Path):
    return sorted((str(p) for p in path.glob("**/*") if (p / "samples.json").exists()), key=natsort_keygen())


class ActivationsExtractorWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Bool(False, help="Dummy trait to trigger observers on changes.")
    activations: Activations | None = None

    def __init__(self, out: widgets.Output | None = None, search_path: str | Path = ".", **kwargs):
        self.search_path = Path(search_path)
        models = list(MODEL_REGISTRY.keys())
        datasets = list(DATASET_REGISTRY.keys())

        # Set defaults
        model = kwargs.get("model", models[0])
        dataset = kwargs.get("dataset", datasets[0])
        max_samples = kwargs.get("max_samples", 200)
        include_response = kwargs.get("include_response", False)
        apply_chat_template = kwargs.get("apply_chat_template", False)

        # Create widgets
        self.w_model = widgets.Dropdown(options=models, value=model, description="Model")
        self.w_dataset = widgets.Dropdown(options=datasets, value=dataset, description="Dataset")
        self.w_max_samples = widgets.IntText(value=max_samples, min=1, description="Max samples")
        self.w_include_response = widgets.Dropdown(
            options=[
                ("none", False),
                ("dataset", True),
                ("custom", "custom"),
                ('"Sorry"', "Sorry"),
                ('"Sure"', "Sure"),
            ],
            value=include_response,
            description="Response",
        )
        self.w_custom_response = widgets.Text(value="", description="Custom")
        self.w_apply_chat_template = widgets.Checkbox(value=apply_chat_template, description="Apply chat template")
        self.btn_extract_activations = widgets.Button(description="Extract Activations", button_style="primary")
        col_extract_activations = widgets.VBox(
            [
                self.w_model,
                self.w_dataset,
                self.w_max_samples,
                self.w_include_response,
                self.w_custom_response,
                self.w_apply_chat_template,
                self.btn_extract_activations,
            ]
        )

        self.w_topk_model = widgets.Dropdown(options=models, value=model, description="Model")
        self.w_topk_layer = widgets.SelectMultiple(description="Layers")
        self.w_topk_k = widgets.IntText(value=10, min=1, description="k")
        self.btn_extract_topk = widgets.Button(
            description="Extract top-k tokens", button_style="primary", disabled=True
        )
        col_extract_topk = widgets.VBox([self.w_topk_model, self.w_topk_layer, self.w_topk_k, self.btn_extract_topk])

        self.w_path = widgets.Combobox(options=get_activation_folders(self.search_path), description="Path")
        self.btn_load = widgets.Button(description="Load Activations", button_style="primary")
        self.btn_save = widgets.Button(description="Save Activations", button_style="success", disabled=True)
        col_load = widgets.VBox(
            [
                self.w_path,
                widgets.HBox([self.btn_load, self.btn_save]),
            ]
        )

        children = [widgets.HBox([col_extract_activations, col_extract_topk]), col_load]
        if out is None:
            self.out = widgets.Output()
            children.append(self.out)
        else:
            self.out = out
        super().__init__(children=children)

        # Register handlers
        self.w_include_response.observe(self._update_custom_response, names="value")
        self._update_custom_response()

        self.btn_extract_activations.on_click(self._do_extract_activations)
        self.btn_save.on_click(self._do_save)
        self.btn_load.on_click(self._do_load)
        self.btn_extract_topk.on_click(self._do_extract_topk)

    def _update_value(self, activations: Activations | None):
        self.activations = activations
        self.value = not self.value

        # Update topk layer options
        if activations is not None:
            self.w_topk_layer.options = activations.layers
            self.w_topk_layer.value = ()
        else:
            self.w_topk_layer.options = []
            self.w_topk_layer.value = ()

    def _update_custom_response(self, *args):
        if self.w_include_response.value == "custom":
            self.w_custom_response.layout.display = None  # show the widget
        else:
            self.w_custom_response.layout.display = "none"  # hide the widget

    @contextlib.contextmanager
    def _update_buttons(self):
        self.btn_extract_activations.disabled = True
        self.btn_save.disabled = True
        self.btn_load.disabled = True
        self.btn_extract_topk.disabled = True
        try:
            yield
        finally:
            self.btn_extract_activations.disabled = False
            self.btn_save.disabled = self.activations is None
            self.btn_load.disabled = False
            self.btn_extract_topk.disabled = self.activations is None

    def _do_extract_activations(self, *args):
        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            # Extract activations
            model, tokenizer = load_model_and_tokenizer(self.model_name)
            dataset = load_dataset_from_spec(self.dataset_name, max_samples=self.max_samples)
            activations = extract_activations(
                model,
                tokenizer,
                dataset,
                include_response=self.include_response,
                apply_chat_template=self.apply_chat_template,
            )

            # Update state
            self._update_value(activations)

            # Update path with default path
            dataset_name = self.dataset_name.replace("/", "_")
            if self.max_samples is not None:
                dataset_name += f"-{self.max_samples}"
            self.w_path.value = str(self.search_path / dataset_name / self.model_name)

    def _do_load(self, *args):
        if not self.path:
            return

        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            path = Path(self.path)
            if not path.exists():
                print(f"Path '{path}' does not exist. Please choose a valid path.")
                return

            # Load activations
            activations = Activations.load(path)
            print(f"Loaded activations: {path}")

            # Update state
            self._update_value(activations)

    def _do_save(self, *args):
        if not self.path or self.activations is None:
            return

        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            path = Path(self.path)
            if path.exists():
                print(f"Path '{path}' already exists. Please choose a different path.")
                return

            # Save activations
            self.activations.save(path)
            print(f"Saved activations: {path}")

            # Update load options
            self.w_path.options = get_activation_folders(self.search_path)

    def _do_extract_topk(self, *args):
        if self.activations is None:
            return

        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            # Extract topk tokens
            model, tokenizer = load_model_and_tokenizer(self.w_topk_model.value)
            for layer in self.w_topk_layer.value:
                self.activations.extract_topk(model, tokenizer, layer=layer, k=self.w_topk_k.value)

    @property
    def model_name(self) -> str:
        return self.w_model.value

    @property
    def dataset_name(self) -> str:
        return self.w_dataset.value

    @property
    def max_samples(self) -> int | None:
        if self.w_max_samples.value > 0:
            return self.w_max_samples.value
        else:
            return None

    @property
    def include_response(self) -> bool | str:
        if self.w_include_response.value == "custom":
            return self.w_custom_response.value
        else:
            return self.w_include_response.value

    @property
    def apply_chat_template(self) -> bool:
        return self.w_apply_chat_template.value

    @property
    def path(self) -> str:
        return self.w_path.value


class ActivationsSelectorWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Bool(False, help="Dummy trait to trigger observers on changes.")

    def __init__(self, **kwargs):
        # Set defaults
        pool_method = kwargs.get("pool_method", "slice")
        pool_index = kwargs.get("pool_index", "-1")
        pool_slice = kwargs.get("pool_slice", "-1::")
        exclude_bos = kwargs.get("exclude_bos", True)
        exclude_special_tokens = kwargs.get("exclude_special_tokens", True)

        # Create widgets
        self.w_safe = widgets.SelectMultiple(description="Safe samples")
        self.w_unsafe = widgets.SelectMultiple(description="Unsafe samples")
        self.w_pool_method = widgets.Dropdown(
            options=["all", "indices", "slice", "first", "mid", "last", "mean"], value=pool_method, description="Tokens"
        )
        self.w_pool_indices = widgets.Text(value=pool_index, description="Indices")
        self.w_pool_slice = widgets.Text(value=pool_slice, description="Slice")
        self.w_exclude_bos = widgets.Checkbox(value=exclude_bos, description="Exclude BOS token")
        self.w_exclude_special_tokens = widgets.Checkbox(
            value=exclude_special_tokens, description="Exclude special tokens"
        )

        super().__init__(
            children=[
                self.w_safe,
                self.w_unsafe,
                self.w_pool_method,
                self.w_pool_indices,
                self.w_pool_slice,
                self.w_exclude_bos,
                self.w_exclude_special_tokens,
            ]
        )

        # Register handlers
        self.w_pool_method.observe(self._update_pool_method, names="value")
        self._update_pool_method()

        self.w_exclude_special_tokens.observe(self._update_value, names="value")
        for w in [
            self.w_safe,
            self.w_unsafe,
            self.w_pool_method,
            self.w_pool_indices,
            self.w_pool_slice,
            self.w_exclude_bos,
            self.w_exclude_special_tokens,
        ]:
            w.observe(self._update_value, names="value")

    def _update_value(self, *args):
        self.value = not self.value

    def _update_pool_method(self, *args):
        if self.w_pool_method.value == "slice":
            self.w_pool_slice.layout.display = None  # show the widget
        else:
            self.w_pool_slice.layout.display = "none"  # hide the widget
        if self.w_pool_method.value == "indices":
            self.w_pool_indices.layout.display = None  # show the widget
        else:
            self.w_pool_indices.layout.display = "none"  # hide the widget

    def set_activations(self, activations: Activations):
        samples_safe = activations.samples[activations.samples["is_safe"]].index.tolist()
        samples_unsafe = activations.samples[~activations.samples["is_safe"]].index.tolist()
        self.w_safe.options = [
            (f"{sample_idx + 1}: {sample_id}", sample_id) for sample_idx, sample_id in enumerate(samples_safe)
        ]
        self.w_unsafe.options = [
            (f"{sample_idx + 1}: {sample_id}", sample_id) for sample_idx, sample_id in enumerate(samples_unsafe)
        ]
        self.w_safe.value = ()
        self.w_unsafe.value = ()

    @property
    def samples(self) -> list[str]:
        return list(self.w_safe.value) + list(self.w_unsafe.value)

    @property
    def pool_method(self) -> PoolMethod:
        if self.w_pool_method.value == "indices":
            return list(map(int, self.w_pool_indices.value.split(",")))
        elif self.w_pool_method.value == "slice":
            return _str_to_slice(self.w_pool_slice.value)
        else:
            return self.w_pool_method.value

    @property
    def exclude_bos(self) -> bool:
        return self.w_exclude_bos.value

    @property
    def exclude_special_tokens(self) -> bool | list[str]:
        return self.w_exclude_special_tokens.value


class TokenEmbeddingsLoaderWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Bool(False, help="Dummy trait to trigger observers on changes.")
    token_embeddings: tuple[list[str], np.array] | None = None

    def __init__(self, out: widgets.Output | None = None, **kwargs):
        models = list(MODEL_REGISTRY.keys())

        # Set defaults
        model = kwargs.get("model", models[0])

        # Create widgets
        self.w_model = widgets.Dropdown(options=models, value=model, description="Model")
        self.w_max_tokens = widgets.IntText(value=10000, min=1, description="Max tokens")
        self.btn_extract = widgets.Button(description="Load token embeddings", button_style="primary")
        self.btn_clear = widgets.Button(description="Clear token embeddings", button_style="danger")

        children = [self.w_model, self.w_max_tokens, widgets.HBox([self.btn_extract, self.btn_clear])]
        if out is None:
            self.out = widgets.Output()
            children.append(self.out)
        else:
            self.out = out
        super().__init__(children=children)

        # Register handlers
        self.btn_extract.on_click(self._do_extract)
        self.btn_clear.on_click(self._do_clear)

    def _update_value(self, token_embeddings: tuple[list[str], np.array] | None):
        self.token_embeddings = token_embeddings
        self.value = not self.value

    @contextlib.contextmanager
    def _update_buttons(self):
        self.btn_extract.disabled = True
        try:
            yield
        finally:
            self.btn_extract.disabled = False

    def _do_extract(self, *args):
        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            model, tokenizer = load_model_and_tokenizer(self.model_name)
            token_ids = select(range(tokenizer.vocab_size), at_most=self.max_tokens)
            token_embeddings = get_token_embeddings(model, tokenizer, token_ids=token_ids)
            self._update_value(token_embeddings)

    def _do_clear(self, *args):
        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            self._update_value(None)
            print("Cleared token embeddings.")

    @property
    def model_name(self) -> str:
        return self.w_model.value

    @property
    def max_tokens(self) -> int:
        return self.w_max_tokens.value
