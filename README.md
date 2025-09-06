# perceptron-playground

[![CI](https://github.com/grthomson/perceptron-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/grthomson/perceptron-playground/actions/workflows/ci.yml)

Experiments with perceptrons with a view to entity resolution as binary classification. Currently running a simple single-layer model on the Iris dataset, to be extended to multilayer versions. Includes plotting, comparison, and research playground. Based on 

## Getting started

Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Acknowledgements

This project is heavily based on examples in  
[**Python Machine Learning, 3rd Edition** by Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book-3rd-edition).
