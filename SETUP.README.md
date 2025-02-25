# Create a virtual environment
```
python3 -m venv .wordle_env
```

# Activate the environment (Linux/macOS)
```
source .wordle_env/bin/activate
```

# Install the requests libraries
```
pip install -r requirements.txt
```

# Run model training
```
python train_model_main.py
```

# Run wordle puzzle
```
python play_wordle_puzzle_main.py
```

# Run all unit tests
```python run_tests.py```

# Run codes coverages
* **Run all the tests in your project and generate a .coverage file containing the coverage data**
```
coverage run -m unittest discover tests
```
* **Generate a detailed report in the terminal**
```
coverage report -m
```
**or**
* **Generate an HTML report**
```
coverage html
```
