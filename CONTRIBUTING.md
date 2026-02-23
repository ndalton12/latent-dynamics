# Contributing  to latent-dynamics

## Prerequisites

- Python 3.13 or higher
- [uv package manager](https://docs.astral.sh/uv/) (install via the official installer, e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh` and follow the prompts)

## Development Setup

1. **Install uv** following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Create the virtual environment and install project dependencies**:
   ```bash
   uv sync
   ```

   This will:
   - Resolve dependencies declared in `pyproject.toml` / `uv.lock`
   - Create a `.venv` folder at the project root (managed by uv)

## Editor Configuration

### VS Code Setup with Ruff

To configure VS Code to format your Python code on save using Ruff via `uvx ruff format`:

1. **Install Ruff and uvx**:
   ```bash
   uv tool install ruff@latest
   ```

2. **Install the Ruff VS Code Extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Ruff" and install the extension by `charliermarsh`

3. **Configure VS Code Settings**:
   The `.vscode/settings.json` is already setup properly, but you can edit it if need be.

This configuration will automatically format your Python code and organize imports every time you save a file, using Ruff via `uvx ruff format`.

## Development Workflow

### Common Commands

Use `uv sync` to keep the local environment in sync with the lockfile and `uvx` for project tooling:

```bash
uv sync --locked            # Install project dependencies exactly as pinned in uv.lock
uvx pytest -n auto -vv      # Run the full test suite
uvx ruff check .            # Run linting without fixing
uvx ruff format .           # Apply formatting
uvx pre-commit run --all-files  # Execute every pre-commit hook locally
uv run python -m reveng      # Execute the package entry point (example)
```

### Running Scripts with uv

- `uv run python path/to/script.py`: Run a script that lives inside the repository using the project environment.
- `uv run -m package.module`: Execute a module as if with `python -m`.
- `uv tool run <command>` / `uvx <command>`: Invoke ad-hoc tooling (for example, `uvx rich-cli tree`).
- Add `--` to forward arguments to the script, e.g. `uv run python scripts/train.py --epochs 10`.

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: For linting and code formatting (configured in `pyproject.toml`)
- **Pytest**: For running tests with parallel execution support

### Adding New Dependencies

1. Activate the environment if needed: `source .venv/bin/activate`.
2. Add the package with uv:
   ```bash
   uv add <package-name>
   ```
   Use `--dev` to add development-only dependencies, and `--group <name>` to target an optional dependency group.
3. Regenerate the lockfile to capture the new requirement:
   ```bash
   uv lock
   ```
4. Re-sync the environment so the dependencies are installed locally:
   ```bash
   uv sync
   ```
5. Commit the updated `pyproject.toml` and `uv.lock` together.

## Package Management

To update dependencies:

1. **Modify `pyproject.toml`** with new dependencies or version constraints
2. **Refresh the lockfile**:
   ```bash
   uv lock --upgrade
   ```
   Use `uv lock --upgrade-package <name>` for targeted upgrades.
3. **Install the updated dependencies locally**:
   ```bash
   uv sync
   ```
4. Commit both `pyproject.toml` and `uv.lock`.
