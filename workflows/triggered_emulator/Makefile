DATA = /Users/noah/data/dev/2020-10-16-emulation-data

all: diagnostics/regressor.pkl diagnostics/regressor_all.pkl

nn.pkl:
	python train_classifier.py notebooks/output/ nn.pkl

regressor.pkl:
	python train_regressor.py --only-triggered --n-estimators 10 --max-depth 10 $(DATA) $@

regressor_all.pkl:
	python train_regressor.py --n-estimators 10 --max-depth 10 $(DATA) $@

diagnostics/%: %
	mkdir -p $@
	python evaluate.py nn.pkl $* $@ || rm -rf $@

upload: nn.pkl regressor.pkl
	# gsutil cp *.pkl gs://vcm-ml-archive/noah/emulator/$(shell date +%F)-triggered-regressor/
	gsutil rsync -r diagnostics gs://vcm-ml-public/noah/emulator/$(shell date +%F)-triggered-regressor/diagnostics



