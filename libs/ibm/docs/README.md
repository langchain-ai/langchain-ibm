# 🧭 Documentation Setup

This directory contains the [MkDocs](https://www.mkdocs.org/) configuration for building and serving the **langchain-ibm** documentation site.

## ⚙️ Installation

Install the documentation dependencies:

- Using uv (recommended)
```bash
uv sync --group docs
```

- Using pip (development install)
```bash
pip install -e ".[docs]"
```

## 🚀 Local Development

To start a local development server:

```bash
mkdocs serve
```

Then open your browser to http://127.0.0.1:8000.

## 🏗️ Build Static Docs

To build the static documentation site:

```bash
mkdocs build
```

The generated site will be available in the `site/` directory.
