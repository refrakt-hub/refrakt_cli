# Contributing to refrakt_cli

Thank you for your interest in contributing! We welcome PRs, issues, and suggestions.

## 🚦 How to Contribute

1. **Fork** this repository and clone your fork.
2. **Create a new branch** for your feature or bugfix.
3. **Write code** and **add tests** for any new feature or fix.
4. **Format and lint** your code (see below).
5. **Ensure all tests pass**.
6. **Open a Pull Request** with a clear description.

---

## 🧹 Formatting & Linting Requirements

We enforce strict code quality. Please make sure your code passes the following checks **before submitting a PR**:

- **pylint**: Score must be **above 9.3** for all files.
- **ruff**: All checks must pass (no errors/warnings).
- **black**: Code must be formatted with black.
- **isort**: Imports must be sorted with isort.
- **mypy**: No type errors.

### Install the Tools

```bash
pip install ruff pylint black mypy isort
```

### Run the Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with ruff
ruff check src/ tests/

# Lint with pylint (score must be >9.3)
pylint src/refrakt_cli/ tests/

# Type check
mypy src/ tests/
```

---

## 🧪 Testing

- **Every module must have a corresponding test file** in the `tests/` directory.
- Run all tests with:

```bash
pytest tests/
```

All tests must pass before submitting a PR.

---

## 🛠️ Updating Dev Dependencies

Dev dependencies are managed in `pyproject.toml` under the `[project.optional-dependencies.dev]` section.

- To add a new dev dependency:
  1. Add it to the `pyproject.toml` file.
  2. Run `pip install -e .[dev]` to install all dev dependencies.

---

## 💡 Tips
- Keep PRs focused and small.
- Write clear commit messages.
- Add docstrings and comments where helpful.
- Be kind and constructive in code reviews.

---

Thank you for helping make refrakt_cli better! 🚀 