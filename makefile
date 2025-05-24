.PHONY: docs

docs: 
	marimo export html-wasm sootopolis_notebook.py --output docs/index.html --mode run

clean:
	rm -r */__pycache__/