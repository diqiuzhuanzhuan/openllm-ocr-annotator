set shell := ["zsh", "-cu"]

export UV_CACHE_DIR := ".uv-cache"

default:
    @just --list

# Install project and development dependencies.
install:
    uv sync --group dev

# Run the test suite. Extra pytest arguments are supported.
test *args:
    uv run pytest {{args}}

# Run Ruff lint checks without modifying files.
lint:
    uv run ruff check .

# Format Python files with Ruff.
fmt:
    uv run ruff format .

# Check formatting without modifying files.
fmt-check:
    uv run ruff format --check .

# Run all checks used before submitting a change.
check: lint fmt-check test

# Run the complete quality-assurance suite.
qa: check

# Run all configured pre-commit hooks.
pre-commit:
    uv run pre-commit run --all-files

# Remove local test, lint, and build caches.
clean:
    rm -rf .pytest_cache .ruff_cache .uv-cache build dist
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
