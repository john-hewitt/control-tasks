# Summarizes the experimental results for the paper
# Designing and Interpreting Probes with Control Tasks

from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('linguistic_task', help='dep or pos')
argp.add_argument('is_control_task', help='0 or 1')
argp.add_argument('hyperparameter', help='rank or sample or wd or dropout or gradient_steps')
args = argp.parse_args()

path_prefix = 'control-tasks/example/config/emnlp19-codalab/'
task = args.linguistic_task
is_control = args.is_control_task
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
print(model_x_axis + ' = ' + str(hyperparameter_options))
for model_type in model_types:
  results_for_model = []
  for hyperparam_value in hyperparameter_options:
    experiment_name = 'linguistic-control_{}_{}_ptb-{}-c{}-{}{}-{}-ELMo1.yaml'.format(task, hyperparam, task, is_control, hyperparam, hyperparam_value, model_type)
    results_path = '{}/{}/dev.label_acc'.format(experiment_name,experiment_name)
    #result = open(results_path).read().strip()
    result = '0.1'
    results_for_model.append(result)

  model_param_name = '{}_c{}_elmo1_{}'.format(hyperparam, is_control, model_type)
  print(model_param_name + ' = ' + str(results_for_model))


