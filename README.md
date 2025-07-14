# refrakt_cli

[![PyPI version](https://img.shields.io/pypi/v/refrakt_cli.svg?logo=pypi)](https://pypi.org/project/refrakt_cli/)

**refrakt_cli** is a command-line interface for the [refrakt_core](https://github.com/refrakt-hub/refrakt_core) deep learning and machine learning research framework. It enables rapid, reproducible, and flexible pipeline training by converting CLI arguments into YAML-based experiment configurations, seamlessly integrating with refrakt_core's modular system.

---

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install refrakt_cli
```

---

## 🚀 What is refrakt_cli?

- A CLI tool to launch, manage, and override ML/DL training pipelines using YAML configs.
- Bridges user-friendly command-line workflows with the powerful abstractions of refrakt_core.
- Supports dynamic hyperparameter overrides, modular experiment configs, and robust logging.

---

## ⚙️ Setup

```bash
git clone https://github.com/refrakt-hub/refrakt_cli.git
cd refrakt_cli

# (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install .
```

---

## 🧪 Usage: Running Experiments

### Run with a config file

```bash
refrakt --config path/to/your_config.yaml
```

### Override hyperparameters on-the-fly

```bash
refrakt --config path/to/your_config.yaml model.name=ResNet optimizer.lr=0.0005 trainer.epochs=20
```

### Supported CLI Flags

| Flag         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `--config`   | Path to YAML config file                                 |
| `--log_type` | Logging backend: `tensorboard`, `wandb`, or both         |
| `--debug`    | Enable debug mode with extra verbosity                   |

---

## 🧩 Project Structure

```
src/refrakt_cli/
├── cli.py                # Main CLI entry point
├── helpers/              # Argument parsing, config overrides, pipeline orchestration
│   ├── argument_parser.py
│   ├── config_overrides.py
│   └── pipeline_orchestrator.py
├── hooks/                # Custom hooks (e.g., hyperparameter overrides)
│   └── hyperparameter_override.py
├── utils/                # Utility functions for pipeline management
│   └── pipeline_utils.py
└── __init__.py
```

- **cli.py**: Main CLI logic and entry point.
- **helpers/**: Argument parsing, config override logic, and pipeline orchestration.
- **hooks/**: Custom hooks for advanced config/hyperparameter handling.
- **utils/**: Utility functions for pipeline and config management.

Test files are located in the `tests/` directory, mirroring the main module structure.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started and best practices for contributing to refrakt_cli.

---

## 📄 License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

---

## 👤 Credits

**Akshath Mangudi**

If you find issues, raise them. If you learn from this, share it.
Built with love and curiosity :)
