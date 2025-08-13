To use the repo:
conda env create -f environment.yml
# or
pip install -r requirements.txt

# formatting:
    ## Install tools
    pip install black ruff

    ## Format code
    black src/...

    ## Lint and fix automatically
    ruff check --fix src/...