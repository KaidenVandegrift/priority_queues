These files require python 3.10+ and matplotlib. After isntalling python 3.10+, you can install matplotlib using "pip install matplotlib". After this, you can use "python test_correctness.py" to run unit tests before running either "python run_experiments.py --quick" or "python run_experiments.py" depending on how long you have. Quick takes roughly 2-3 minutes, the full test takes 15-20 minutes or even longer. After running experiments, you can generate the .png plots with "python analyze_results.py". After analyzing results, you can run the metrics with "python run_metrics.py --quick" or "python run_metrics.py". You can then generate more plots with "python plot_metrics.py". The running order should be the following:
	python test_correctness.py
	python run_experiments.py [--quick]
	python analyze_results.py
	python run_metrics.py [--quick]
	python plot_metrics.py
I did not include any of the generated files here, they are instead used within my report.