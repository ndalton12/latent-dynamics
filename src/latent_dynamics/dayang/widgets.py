import contextlib
from pathlib import Path

import ipywidgets as widgets
from natsort import natsort_keygen
from traitlets import Bool

from latent_dynamics.dayang.activations import Activations, PoolMethod, extract_activations
from latent_dynamics.dayang.data import DATASET_REGISTRY, load_dataset_from_spec
from latent_dynamics.dayang.model import MODEL_REGISTRY, load_model_and_tokenizer


def _str_to_slice(s: str) -> slice | None:
    return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(":")))


def get_activation_folders(path: Path):
    return sorted((str(p) for p in path.glob("**/*") if (p / "metadata.json").exists()), key=natsort_keygen())


class ActivationsExtractorWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Bool(False, help="Dummy trait to trigger observers on changes.")
    activations: Activations | None = None

    def __init__(self, search_path: str | Path = ".", **kwargs):
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
        self.w_max_samples = widgets.IntText(value=max_samples, min=1, description="Max samples:")
        self.w_include_response = widgets.Dropdown(
            options=[
                ("none", False),
                ("dataset", True),
                ('"Sorry"', "Sorry"),
                ('"Sure"', "Sure"),
                ("custom", "custom"),
            ],
            value=include_response,
            description="Response",
        )
        self.w_custom_response = widgets.Text(value="", description="Custom")
        self.w_apply_chat_template = widgets.Checkbox(value=apply_chat_template, description="Apply chat template")
        self.btn_extract = widgets.Button(description="Extract Activations", button_style="primary")
        col_extract = widgets.VBox(
            [
                self.w_model,
                self.w_dataset,
                self.w_max_samples,
                self.w_include_response,
                self.w_custom_response,
                self.w_apply_chat_template,
                self.btn_extract,
            ]
        )

        self.w_path = widgets.Combobox(options=get_activation_folders(self.search_path), description="Path")
        self.btn_load = widgets.Button(description="Load Activations", button_style="primary")
        self.btn_save = widgets.Button(description="Save Activations", button_style="success", disabled=True)
        col_load = widgets.VBox(
            [
                self.w_path,
                widgets.HBox([self.btn_load, self.btn_save]),
            ]
        )

        self.out = widgets.Output()
        super().__init__(children=[widgets.HBox([col_extract, col_load]), self.out])

        # Register handlers
        self.w_include_response.observe(self._update_custom_response, names="value")
        self._update_custom_response()

        self.btn_extract.on_click(self._do_extract)
        self.btn_save.on_click(self._do_save)
        self.btn_load.on_click(self._do_load)

    def _update_value(self, activations: Activations | None):
        self.activations = activations
        self.value = not self.value

    def _update_custom_response(self, *args):
        if self.w_include_response.value == "custom":
            self.w_custom_response.layout.display = None  # show the widget
        else:
            self.w_custom_response.layout.display = "none"  # hide the widget

    @contextlib.contextmanager
    def _update_buttons(self):
        self.btn_extract.disabled = True
        self.btn_save.disabled = True
        self.btn_load.disabled = True
        try:
            yield
        finally:
            self.btn_extract.disabled = False
            self.btn_save.disabled = self.activations is None
            self.btn_load.disabled = False

    def _do_extract(self, *args):
        with self._update_buttons(), self.out:
            self.out.clear_output(wait=True)

            # Extract activations
            model, tokenizer = load_model_and_tokenizer(self.model)
            dataset = load_dataset_from_spec(self.dataset, max_samples=self.max_samples)
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
            dataset_name = self.w_dataset.value.replace("/", "_")
            if self.max_samples is not None:
                dataset_name += f"-{self.max_samples}"
            self.w_path.value = str(self.search_path / dataset_name / self.w_model.value)

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

    @property
    def model(self) -> str:
        return self.w_model.value

    @property
    def dataset(self) -> str:
        return self.w_dataset.value

    @property
    def max_samples(self) -> int | None:
        if self.w_max_samples.value > 0:
            return self.w_max_samples.value
        else:
            return None

    @property
    def include_response(self) -> bool | str:
        if self.w_include_response.value == "Custom":
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
        pool = kwargs.get("pool", "slice")
        pool_slice = kwargs.get("pool_slice", "-1::")
        exclude_bos = kwargs.get("exclude_bos", True)
        exclude_special = kwargs.get("exclude_special", True)

        # Create widgets
        self.w_safe = widgets.SelectMultiple(description="Safe samples")
        self.w_unsafe = widgets.SelectMultiple(description="Unsafe samples")
        self.w_pool = widgets.Dropdown(
            options=["all", "first", "mid", "last", "mean", "slice"], value=pool, description="Tokens"
        )
        self.w_pool_slice = widgets.Text(value=pool_slice, description="Slice")
        self.w_exclude_bos = widgets.Checkbox(value=exclude_bos, description="Exclude BOS token")
        self.w_exclude_special = widgets.Checkbox(value=exclude_special, description="Exclude special tokens")

        super().__init__(
            children=[
                self.w_safe,
                self.w_unsafe,
                self.w_pool,
                self.w_pool_slice,
                self.w_exclude_bos,
                self.w_exclude_special,
            ]
        )

        # Register handlers
        self.w_pool.observe(self._update_pool_slice, names="value")
        self._update_pool_slice()

        self.w_exclude_special.observe(self._update_value, names="value")
        for w in [
            self.w_safe,
            self.w_unsafe,
            self.w_pool,
            self.w_pool_slice,
            self.w_exclude_bos,
            self.w_exclude_special,
        ]:
            w.observe(self._update_value, names="value")

    def _update_value(self, *args):
        self.value = not self.value

    def _update_pool_slice(self, *args):
        if self.w_pool.value == "slice":
            self.w_pool_slice.layout.display = None  # show the widget
        else:
            self.w_pool_slice.layout.display = "none"  # hide the widget

    def set_activations(self, activations: Activations):
        samples_safe = activations.metadata[activations.metadata["is_safe"]].index.tolist()
        samples_unsafe = activations.metadata[~activations.metadata["is_safe"]].index.tolist()
        self.w_safe.options = samples_safe
        self.w_unsafe.options = samples_unsafe
        self.w_safe.value = ()
        self.w_unsafe.value = ()

    @property
    def samples(self) -> list[str]:
        return list(self.w_safe.value) + list(self.w_unsafe.value)

    @property
    def pool_method(self) -> PoolMethod:
        if self.w_pool.value == "slice":
            return _str_to_slice(self.w_pool_slice.value)
        else:
            return self.w_pool.value

    @property
    def exclude_bos(self) -> bool:
        return self.w_exclude_bos.value

    @property
    def exclude_special_tokens(self) -> bool | list[str]:
        return self.w_exclude_special.value
