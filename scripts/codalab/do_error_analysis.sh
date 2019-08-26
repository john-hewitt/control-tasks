python3.6 scripts/codalab/aggregate_predictions.py \
  linear.predictions \
  ptb3-wsj-dev.conllx \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run1.yaml/error_analysis_linear_run_1/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run2.yaml/error_analysis_linear_run_2/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run3.yaml/error_analysis_linear_run_3/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run4.yaml/error_analysis_linear_run_4/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run5.yaml/error_analysis_linear_run_5/dev.predictions

python3.6 scripts/codalab/aggregate_predictions.py \
  mlp.predictions \
  ptb3-wsj-dev.conllx \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run1.yaml/error_analysis_mlp_run_1/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run2.yaml/error_analysis_mlp_run_2/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run3.yaml/error_analysis_mlp_run_3/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run4.yaml/error_analysis_mlp_run_4/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run5.yaml/error_analysis_mlp_run_5/dev.predictions 

python3.6 scripts/codalab/plot_examples.py \
  ptb3-wsj-dev.conllx \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-1hid-ELMo1-run1.yaml/error_analysis_mlp_run_1/dev.predictions \
  example-config-emnlp19-codalab-error-analysis-ptb-pos-c0-rank1000-0hid-ELMo1-run1.yaml/error_analysis_linear_run_1/dev.predictions

python3.6 scripts/codalab/error_analysis.py  \
  mlp.predictions \
  linear.predictions
