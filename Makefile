PY ?= python
EPOCHS ?= 10
OPTS := sgd momentum rmsprop adam

.PHONY: install test smoke train-mnist train-cifar benchmark plots cost all clean

install:
	$(PY) -m pip install -e ".[torch,dev]"

test:
	$(PY) -m pytest -q

smoke:  ## offline end-to-end check on synthetic data
	for fw in numpy torch; do for a in mlp cnn; do \
	  $(PY) scripts/train.py --framework $$fw --arch $$a --synthetic --epochs 1 --out results/smoke; done; done

train-mnist:
	for fw in numpy torch; do for a in mlp cnn; do for o in $(OPTS); do \
	  $(PY) scripts/train.py --framework $$fw --arch $$a --dataset mnist --optimizer $$o --epochs $(EPOCHS); done; done; done

train-cifar:
	for fw in numpy torch; do for a in mlp cnn; do for o in $(OPTS); do \
	  $(PY) scripts/train.py --framework $$fw --arch $$a --dataset cifar10 --optimizer $$o --epochs $(EPOCHS); done; done; done

benchmark:
	$(PY) scripts/benchmark.py
	$(PY) scripts/benchmark.py --threads 1 --no-cuda --out results/single_thread

plots:
	$(PY) scripts/plot_results.py

cost:
	$(PY) scripts/cost_report.py

all: test train-mnist train-cifar benchmark plots cost

clean:
	rm -rf results/runs results/figures results/smoke results/single_thread results/*.csv results/benchmark.md
