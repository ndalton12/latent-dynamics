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
            options=[False, True, "Sorry", "Sure"], value=include_response, description="Include response"
        )
        self.w_apply_chat_template = widgets.Checkbox(value=apply_chat_template, description="Apply chat template")
        self.btn_extract = widgets.Button(description="Extract Activations", button_style="primary")
        col_extract = widgets.VBox(
            [
                self.w_model,
                self.w_dataset,
                self.w_max_samples,
                self.w_include_response,
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
        self.btn_extract.on_click(self.do_extract)
        self.btn_save.on_click(self.do_save)
        self.btn_load.on_click(self.do_load)

    def __update_value(self, activations: Activations | None):
        self.activations = activations
        self.value = not self.value

    @contextlib.contextmanager
    def update_buttons(self):
        self.btn_extract.disabled = True
        self.btn_save.disabled = True
        self.btn_load.disabled = True
        try:
            yield
        finally:
            self.btn_extract.disabled = False
            self.btn_save.disabled = self.activations is None
            self.btn_load.disabled = False

    def do_extract(self, *args):
        with self.update_buttons(), self.out:
            self.out.clear_output(wait=True)

            # Extract activations
            model, tokenizer = load_model_and_tokenizer(self.w_model.value)
            dataset = load_dataset_from_spec(self.w_dataset.value, max_samples=self.w_max_samples.value)
            activations = extract_activations(
                model,
                tokenizer,
                dataset,
                include_response=self.w_include_response.value,
                apply_chat_template=self.w_apply_chat_template.value,
            )

            # Update state
            self.__update_value(activations)

            # Update path with default path
            default_path = (
                self.search_path
                / f"{self.w_dataset.value.replace('/', '_')}-{self.w_max_samples.value}"
                / self.w_model.value
            )
            self.w_path.value = str(default_path)

    def do_load(self, *args):
        path = self.w_path.value
        if not path:
            return

        with self.update_buttons(), self.out:
            self.out.clear_output(wait=True)

            path = Path(path)
            if not path.exists():
                print(f"Path '{path}' does not exist. Please choose a valid path.")
                return

            # Load activations
            print(f"Loading activations from '{path}'...")
            activations = Activations.load(path)
            print("Loaded successfully!")

            # Update state
            self.__update_value(activations)

    def do_save(self, *args):
        path = self.w_path.value
        if not path or self.activations is None:
            return

        with self.update_buttons(), self.out:
            self.out.clear_output(wait=True)

            path = Path(path)
            if path.exists():
                print(f"Path '{path}' already exists. Please choose a different path.")
                return

            # Save activations
            print(f"Saving activations to '{path}'...")
            self.activations.save(path)
            print("Saved successfully!")

            # Update load options
            self.w_path.options = get_activation_folders(self.search_path)


class ActivationsSelectorWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Bool(False, help="Dummy trait to trigger observers on changes.")

    def __init__(self, **kwargs):
        # Set defaults
        pool = kwargs.get("pool", "last")
        pool_slice = kwargs.get("pool_slice", "::")
        exclude_bos = kwargs.get("exclude_bos", True)
        exclude_special = kwargs.get("exclude_special", True)

        # Create widgets
        self.w_safe = widgets.SelectMultiple(description="Safe samples")
        self.w_unsafe = widgets.SelectMultiple(description="Unsafe samples")
        self.w_pool = widgets.Dropdown(
            options=["all", "first", "mid", "last", "mean", "slice"], value=pool, description="Tokens"
        )
        self.w_pool_slice = widgets.Text(value=pool_slice, description="Tokens slice")
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
        self.w_pool.observe(self.update_pool_slice, names="value")
        self.update_pool_slice()

        self.w_exclude_special.observe(self.__update_value, names="value")
        for w in [
            self.w_safe,
            self.w_unsafe,
            self.w_pool,
            self.w_pool_slice,
            self.w_exclude_bos,
            self.w_exclude_special,
        ]:
            w.observe(self.__update_value, names="value")

    def __update_value(self, *args):
        self.value = not self.value

    def update_pool_slice(self, *args):
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
