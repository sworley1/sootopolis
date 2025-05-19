.PHONY: docs

docs: 
	marimo export html-wasm sootopolis_notebook.py --output docs/index.html --mode run