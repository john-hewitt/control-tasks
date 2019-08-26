# Prints the results of ELMo1 vs ELMo2 experiments

import sys

result_dict = {}

pos_probes = ['0hid', '1hid']
pos_reprs = ['ProjELMo0', 'ELMo1', 'ELMo2']
for is_control in ['0', '1']:
  for probe_type in pos_probes:
    for repr_type in pos_reprs:
      experiment_name1 = 'example-config-emnlp19-codalab-elmo1-elmo2-pos-ptb-pos-c{}-rank1000-{}-{}.yaml'.format(is_control, probe_type, repr_type)

      experiment_name2 = 'elmo1-elmo2_pos_ptb-pos-c{}-rank1000-{}-{}.yaml'.format(is_control, probe_type, repr_type)
      results_path = '{}/{}/dev.label_acc'.format(experiment_name1, experiment_name2)
      result = float(open(results_path).read().strip())
      result_dict[('pos', is_control, probe_type, repr_type)] = result

dep_probes = ['bilinear', '1hid']
dep_reprs = ['ProjELMo0', 'ELMo1', 'ELMo2']
for is_control in ['0', '1']:
  for probe_type in dep_probes:
    for repr_type in dep_reprs:
      experiment_name1 = 'example-config-emnlp19-codalab-elmo1-elmo2-dep-ptb-dep-c{}-s40000-{}-{}.yaml'.format(is_control, probe_type, repr_type)

      experiment_name2 = 'elmo1-elmo2_dep_ptb-dep-c{}-s40000-{}-{}.yaml'.format(is_control, probe_type, repr_type)
      results_path = '{}/{}/dev.uuas'.format(experiment_name1, experiment_name2)
      result = float(open(results_path).read().strip())
      result_dict[(is_control, probe_type, repr_type)] = result
      result_dict[('dep', is_control, probe_type, repr_type)] = result

print('POS')
print('\tLinear\t\tMLP\t')
print('Model\tAccuracy\tSelectivity\tAccuracy\tSelectivity')
print('\\begin{tabular}{l r r r r r }', file=sys.stderr)
print('\\toprule', file=sys.stderr)
print('\\multicolumn{6}{c}{\\bf Part-of-speech Tagging} \\\\ \\cmidrule{1-6}', file=sys.stderr)
print('& \\multicolumn{2}{c}{Linear} & & \\multicolumn{2}{c}{MLP-1} \\\\ \\cmidrule{2-3} \\cmidrule{5-6}', file=sys.stderr)
print('\\bf Model & Accuracy & Selectivity && Accuracy & Selectivity\\\\', file=sys.stderr)
print('\\midrule', file=sys.stderr)
for repr_type in pos_reprs:
  ling_acc_0hid = result_dict[('pos', '0', '0hid', repr_type)]
  control_acc_0hid = result_dict[('pos', '1', '0hid', repr_type)]
  selectivity_0hid = ling_acc_0hid - control_acc_0hid
  ling_acc_1hid = result_dict[('pos', '0', '1hid', repr_type)]
  control_acc_1hid = result_dict[('pos', '1', '1hid', repr_type)]
  selectivity_1hid = ling_acc_1hid - control_acc_1hid
  print('\t'.join([repr_type] + ['{0:.1f}'.format(100*x) for x in [ling_acc_0hid, selectivity_0hid, ling_acc_1hid, selectivity_1hid]]))
  print(' & '.join([repr_type] + ['${0:.1f}$'.format(100*x) for x in [ling_acc_0hid, selectivity_0hid]] + [''] + ['${0:.1f}$'.format(100*x) for x in [ling_acc_1hid, selectivity_1hid]]) + '\\\\', file=sys.stderr)
print('Dep')
print('\\bottomrule', file=sys.stderr)
print('\\toprule', file=sys.stderr)
print('\\multicolumn{6}{c}{\\bf Dependency Edge Prediction} \\\\ \\cmidrule{1-6}', file=sys.stderr)
print('& \\multicolumn{2}{c}{Bilinear} & & \\multicolumn{2}{c}{MLP-1} \\\\ \\cmidrule{2-3} \\cmidrule{5-6}', file=sys.stderr)
print('\\bf Model & Accuracy & Selectivity && Accuracy & Selectivity\\\\', file=sys.stderr)
print('\\midrule', file=sys.stderr)
for repr_type in dep_reprs:
  ling_acc_0hid = result_dict[('dep', '0', 'bilinear', repr_type)]
  control_acc_0hid = result_dict[('dep', '1', 'bilinear', repr_type)]
  selectivity_0hid = ling_acc_0hid - control_acc_0hid
  ling_acc_1hid = result_dict[('dep', '0', '1hid', repr_type)]
  control_acc_1hid = result_dict[('dep', '1', '1hid', repr_type)]
  selectivity_1hid = ling_acc_1hid - control_acc_1hid
  print('\t'.join([repr_type] + ['{0:.1f}'.format(100*x) for x in [ling_acc_0hid, selectivity_0hid, ling_acc_1hid, selectivity_1hid]]))
  print(' & '.join([repr_type] + ['${0:.1f}$'.format(100*x) for x in [ling_acc_0hid, selectivity_0hid]] + [''] + ['${0:.1f}$'.format(100*x) for x in [ling_acc_1hid, selectivity_1hid]]) + '\\\\', file=sys.stderr)
print('\\bottomrule', file=sys.stderr)
print('\\end{tabular}', file=sys.stderr)
