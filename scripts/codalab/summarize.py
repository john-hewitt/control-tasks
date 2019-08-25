# Summarizes the experimental results for the paper
# Designing and Interpreting Probes with Control Tasks

from argparse import ArgumentParser
from collections import defaultdict
import json

argp = ArgumentParser()
argp.add_argument('linguistic_task', help='dep or pos')
#argp.add_argument('is_control_task', help='0 or 1')
argp.add_argument('hyperparameter', help='rank or sample or wd or dropout or gradient_steps')
args = argp.parse_args()

path_prefix = 'control-tasks/example/config/emnlp19-codalab/'
task = args.linguistic_task
#is_control = args.is_control_task
hyperparam = args.hyperparameter

pos_hyperparameter_to_options = {
    'rank': ['2', '4', '10', '45', '1000'],
    'wd': ['0', '0.01', '0.1', '1'],
    'dropout': ['0', '0.2', '0.4', '0.6', 0.8],
    'sample': ['400', '4000', '40000'],
    'gradient_steps': ['1500', '3000', '6000', '12500', '25000', '50000']
    }
dep_hyperparameter_to_options = {
    'rank': ['5', '10', '50', '100', '1000'],
    'wd': ['0', '0.01', '0.1', '1'],
    'dropout': ['0', '0.2', '0.4', '0.6', 0.8],
    'sample': ['400', '4000', '40000'],
    'gradient_steps': ['1500', '3000', '6000', '12500', '25000', '50000']
    }
hyperparameter_to_options = pos_hyperparameter_to_options if args.linguistic_task == 'pos' else dep_hyperparameter_to_options
hyperparameter_options = hyperparameter_to_options[hyperparam]
model_types = ['0hid', '1hid', '2hid'] if args.linguistic_task == 'pos' else ['bilinear', '1hid', '2hid']

results_path_template = '{}/{}/dev.label_acc' if task == 'pos' else '{}/{}/dev.uuas'

experiments = []
model_x_axis = hyperparam +'_x_axis'
result_dictionary = {hyperparam: {'0':{}, '1':{}}, 'hyperparameter_options': hyperparameter_options}
print(model_x_axis + ' = ' + str([float(x) for x in hyperparameter_options]))
short_hyperparam_name = 'grad' if hyperparam == 'gradient_steps' else hyperparam
short_hyperparam_name = 'drop' if hyperparam == 'dropout' else short_hyperparam_name
short_hyperparam_name = 's' if hyperparam == 'sample' else short_hyperparam_name
for model_type in model_types:
  for is_control in ['0', '1']:
    results_for_model = []
    for hyperparam_value in hyperparameter_options:
      experiment_name1 = 'example-config-emnlp19-codalab-linguistic-control-{}-{}-ptb-{}-c{}-{}{}-{}-ELMo1.yaml'.format(task, hyperparam, task, is_control, short_hyperparam_name, hyperparam_value, model_type)
      experiment_name2 = 'linguistic-control_{}_{}_ptb-{}-c{}-{}{}-{}-ELMo1.yaml'.format(task, hyperparam, task, is_control, short_hyperparam_name, hyperparam_value, model_type)
      results_path = results_path_template.format(experiment_name1,experiment_name2)
      result = float(open(results_path).read().strip())
      results_for_model.append(result)
    result_dictionary[hyperparam][is_control][model_type] = results_for_model
    
    model_param_name = '{}_c{}_elmo1_{}'.format(hyperparam, is_control, model_type)
    print(model_param_name + ' = ' + str(results_for_model))
with open('results.json', 'w') as fout:
  json.dump(result_dictionary, fout)

