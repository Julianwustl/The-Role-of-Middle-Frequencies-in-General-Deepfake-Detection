

train: 
	export TOKENIZERS_PARALLELISM=false && python3 train.py

run-test:

	 python3 model/VisionTransformer/videoMAE.py

test:
	python3 evaluation.py

dataset:
	python3 scripts/create_csv.py