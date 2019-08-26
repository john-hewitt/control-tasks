import json
import sys

# Start of table code
print('\\begin{tabular}{l  c c c | c c c}')
print('\\toprule')
print('\\bf Probe & \\bf PoS & Ctl & \\bf Select. & \\bf Dep & Ctl & \\bf Select.\\\\')
print('\\midrule')

### Probes with default hyperparams
print('\\multicolumn{7}{c}{Probes with ``Default'' Hyperparameters}')
print('\\vspace{3pt}\\\\')

pos_dropout_results = json.load(open('pos-codalab/summarize_pos_dropout/results.json'))
dep_dropout_results = json.load(open('dep-codalab/summarize_dep_dropout/results.json'))

pos_linear_default_acc = pos_dropout_results["dropout"]["0"]['0hid'][0]
pos_linear_default_ctl = pos_dropout_results["dropout"]["1"]['0hid'][0]
pos_linear_default_select = pos_linear_default_acc - pos_linear_default_ctl
#pos_linear_default_select = (1-pos_linear_default_ctl)/ (1-pos_linear_default_acc)
dep_bilinear_default_acc = dep_dropout_results["dropout"]["0"]['bilinear'][0]
dep_bilinear_default_ctl = dep_dropout_results["dropout"]["1"]['bilinear'][0]
dep_bilinear_default_select = dep_bilinear_default_acc - dep_bilinear_default_ctl
#dep_bilinear_default_select = (1-dep_bilinear_default_ctl)/(1- dep_bilinear_default_acc)

pos_1hid_default_acc = pos_dropout_results["dropout"]["0"]['1hid'][0]
pos_1hid_default_ctl = pos_dropout_results["dropout"]["1"]['1hid'][0]
pos_1hid_default_select = pos_1hid_default_acc - pos_1hid_default_ctl
#pos_1hid_default_select = (1-pos_1hid_default_ctl)/(1- pos_1hid_default_acc)
dep_1hid_default_acc = dep_dropout_results["dropout"]["0"]['1hid'][0]
dep_1hid_default_ctl = dep_dropout_results["dropout"]["1"]['1hid'][0]
dep_1hid_default_select = dep_1hid_default_acc - dep_1hid_default_ctl
#dep_1hid_default_select = (1-dep_1hid_default_ctl)/(1-dep_1hid_default_acc)

pos_2hid_default_acc = pos_dropout_results["dropout"]["0"]['2hid'][0]
pos_2hid_default_ctl = pos_dropout_results["dropout"]["1"]['2hid'][0]
pos_2hid_default_select = pos_2hid_default_acc - pos_2hid_default_ctl
#pos_2hid_default_select = (1-pos_2hid_default_ctl) / (1-pos_2hid_default_acc)
dep_2hid_default_acc = dep_dropout_results["dropout"]["0"]['2hid'][0]
dep_2hid_default_ctl = dep_dropout_results["dropout"]["1"]['2hid'][0]
dep_2hid_default_select = dep_2hid_default_acc - dep_2hid_default_ctl
#dep_2hid_default_select = (1-dep_2hid_default_ctl) / (1-dep_2hid_default_acc)

default_linear_tex_line = [pos_linear_default_acc, pos_linear_default_ctl, pos_linear_default_select]
default_linear_tex_line = ' & '.join(['Linear'] + ['${0:.1f}$'.format(100*x) for x in default_linear_tex_line] + ['-', '-', '-']) + '\\\\'
print(default_linear_tex_line)

default_bilinear_tex_line = [dep_bilinear_default_acc, dep_bilinear_default_ctl, dep_bilinear_default_select]
default_bilinear_tex_line = ' & '.join(['Bilinear', '-', '-', '-']+ ['${0:.1f}$'.format(100*x) for x in default_bilinear_tex_line]) + '\\\\'
print(default_bilinear_tex_line)

default_1hid_tex_line = [pos_1hid_default_acc, pos_1hid_default_ctl, pos_1hid_default_select] + [dep_1hid_default_acc, dep_1hid_default_ctl, dep_1hid_default_select]
default_1hid_tex_line = ' & '.join(['MLP-1'] + ['${0:.1f}$'.format(100*x) for x in default_1hid_tex_line]) + '\\\\'
print(default_1hid_tex_line)

default_2hid_tex_line = [pos_2hid_default_acc, pos_2hid_default_ctl, pos_2hid_default_select] + [dep_2hid_default_acc, dep_2hid_default_ctl, dep_2hid_default_select]
default_2hid_tex_line = ' & '.join(['MLP-2'] +['${0:.1f}$'.format(100*x) for x in default_2hid_tex_line]) + '\\\\'
print(default_2hid_tex_line)

print('\\midrule')

### Probes with .4 dropout
print('\\multicolumn{7}{c}{Probes with $0.4$ Dropout}')
print('\\vspace{3pt}\\\\')

pos_dropout_results = json.load(open('pos-codalab/summarize_pos_dropout/results.json'))
dep_dropout_results = json.load(open('dep-codalab/summarize_dep_dropout/results.json'))

pos_linear_default_acc = pos_dropout_results["dropout"]["0"]['0hid'][2]
pos_linear_default_ctl = pos_dropout_results["dropout"]["1"]['0hid'][2]
pos_linear_default_select = pos_linear_default_acc - pos_linear_default_ctl
#pos_linear_default_select = (1-pos_linear_default_ctl)/ (1-pos_linear_default_acc)
dep_bilinear_default_acc = dep_dropout_results["dropout"]["0"]['bilinear'][2]
dep_bilinear_default_ctl = dep_dropout_results["dropout"]["1"]['bilinear'][2]
dep_bilinear_default_select = dep_bilinear_default_acc - dep_bilinear_default_ctl
#dep_bilinear_default_select = (1-dep_bilinear_default_ctl)/(1- dep_bilinear_default_acc)

pos_1hid_default_acc = pos_dropout_results["dropout"]["0"]['1hid'][2]
pos_1hid_default_ctl = pos_dropout_results["dropout"]["1"]['1hid'][2]
pos_1hid_default_select = pos_1hid_default_acc - pos_1hid_default_ctl
#pos_1hid_default_select = (1-pos_1hid_default_ctl)/(1- pos_1hid_default_acc)
dep_1hid_default_acc = dep_dropout_results["dropout"]["0"]['1hid'][2]
dep_1hid_default_ctl = dep_dropout_results["dropout"]["1"]['1hid'][2]
dep_1hid_default_select = dep_1hid_default_acc - dep_1hid_default_ctl
#dep_1hid_default_select = (1-dep_1hid_default_ctl)/(1-dep_1hid_default_acc)

pos_2hid_default_acc = pos_dropout_results["dropout"]["0"]['2hid'][2]
pos_2hid_default_ctl = pos_dropout_results["dropout"]["1"]['2hid'][2]
pos_2hid_default_select = pos_2hid_default_acc - pos_2hid_default_ctl
#pos_2hid_default_select = (1-pos_2hid_default_ctl) / (1-pos_2hid_default_acc)
dep_2hid_default_acc = dep_dropout_results["dropout"]["0"]['2hid'][2]
dep_2hid_default_ctl = dep_dropout_results["dropout"]["1"]['2hid'][2]
dep_2hid_default_select = dep_2hid_default_acc - dep_2hid_default_ctl
#dep_2hid_default_select = (1-dep_2hid_default_ctl) / (1-dep_2hid_default_acc)

default_linear_tex_line = [pos_linear_default_acc, pos_linear_default_ctl, pos_linear_default_select]
default_linear_tex_line = ' & '.join(['Linear'] + ['${0:.1f}$'.format(100*x) for x in default_linear_tex_line] + ['-', '-', '-']) + '\\\\'
print(default_linear_tex_line)

default_bilinear_tex_line = [dep_bilinear_default_acc, dep_bilinear_default_ctl, dep_bilinear_default_select]
default_bilinear_tex_line = ' & '.join(['Bilinear', '-', '-', '-']+ ['${0:.1f}$'.format(100*x) for x in default_bilinear_tex_line]) + '\\\\'
print(default_bilinear_tex_line)

default_1hid_tex_line = [pos_1hid_default_acc, pos_1hid_default_ctl, pos_1hid_default_select] + [dep_1hid_default_acc, dep_1hid_default_ctl, dep_1hid_default_select]
default_1hid_tex_line = ' & '.join(['MLP-1'] +['${0:.1f}$'.format(100*x) for x in default_1hid_tex_line]) + '\\\\'
print(default_1hid_tex_line)

default_2hid_tex_line = [pos_2hid_default_acc, pos_2hid_default_ctl, pos_2hid_default_select] + [dep_2hid_default_acc, dep_2hid_default_ctl, dep_2hid_default_select]
default_2hid_tex_line = ' & '.join(['MLP-2'] +['${0:.1f}$'.format(100*x) for x in default_2hid_tex_line]) + '\\\\'
print(default_2hid_tex_line)

print('\\midrule')

### Probes designed with control tasks
print('\\multicolumn{7}{c}{Probes with Control Tasks }')
print('\\vspace{3pt}\\\\')

pos_rank_results = json.load(open('pos-codalab/summarize_pos_rank/results.json'))
dep_wd_results = json.load(open('dep-codalab/summarize_dep_wd/results.json'))

pos_linear_default_acc = pos_rank_results["rank"]["0"]['0hid'][2] # Rank=10 
pos_linear_default_ctl = pos_rank_results["rank"]["1"]['0hid'][2] # Rank=10
pos_linear_default_select = pos_linear_default_acc - pos_linear_default_ctl
print(pos_rank_results['hyperparameter_options'][2], 'rank=10',file=sys.stderr)
#pos_linear_default_select = (1-pos_linear_default_ctl)/ (1-pos_linear_default_acc)
dep_bilinear_default_acc = dep_wd_results["wd"]["0"]['bilinear'][1] # wd=.01 
dep_bilinear_default_ctl = dep_wd_results["wd"]["1"]['bilinear'][1] # wd=.01
print(dep_wd_results['hyperparameter_options'][1], 'wd=.01',file=sys.stderr)
dep_bilinear_default_select = dep_bilinear_default_acc - dep_bilinear_default_ctl
#dep_bilinear_default_select = (1-dep_bilinear_default_ctl)/(1- dep_bilinear_default_acc)

pos_1hid_default_acc = pos_rank_results["rank"]["0"]['1hid'][3] # rank=45 
pos_1hid_default_ctl = pos_rank_results["rank"]["1"]['1hid'][3] # rank=45
pos_1hid_default_select = pos_1hid_default_acc - pos_1hid_default_ctl
print(pos_rank_results['hyperparameter_options'][3], 'rank=45',file=sys.stderr)
#pos_1hid_default_select = (1-pos_1hid_default_ctl)/(1- pos_1hid_default_acc)
dep_1hid_default_acc = dep_wd_results["wd"]["0"]['1hid'][2] # wd=.1 
dep_1hid_default_ctl = dep_wd_results["wd"]["1"]['1hid'][2] # wd=.1
dep_1hid_default_select = dep_1hid_default_acc - dep_1hid_default_ctl
print(dep_wd_results['hyperparameter_options'][2], 'wd=0.1',file=sys.stderr)
#dep_1hid_default_select = (1-dep_1hid_default_ctl)/(1-dep_1hid_default_acc)

pos_2hid_default_acc = pos_rank_results["rank"]["0"]['2hid'][3] # rank=45 
pos_2hid_default_ctl = pos_rank_results["rank"]["1"]['2hid'][3] # rank=45
pos_2hid_default_select = pos_2hid_default_acc - pos_2hid_default_ctl
print(pos_rank_results['hyperparameter_options'][3], 'rank=45',file=sys.stderr)
#pos_2hid_default_select = (1-pos_2hid_default_ctl) / (1-pos_2hid_default_acc)
dep_2hid_default_acc = dep_wd_results["wd"]["0"]['2hid'][2] # wd=.1 
dep_2hid_default_ctl = dep_wd_results["wd"]["1"]['2hid'][2] # wd=.1
dep_2hid_default_select = dep_2hid_default_acc - dep_2hid_default_ctl
print(dep_wd_results['hyperparameter_options'][2], 'wd=0.1',file=sys.stderr)
#dep_2hid_default_select = (1-dep_2hid_default_ctl) / (1-dep_2hid_default_acc)

default_linear_tex_line = [pos_linear_default_acc, pos_linear_default_ctl, pos_linear_default_select]
default_linear_tex_line = ' & '.join(['Linear'] + ['${0:.1f}$'.format(100*x) for x in default_linear_tex_line] + ['-', '-', '-']) + '\\\\'
print(default_linear_tex_line)

default_bilinear_tex_line = [dep_bilinear_default_acc, dep_bilinear_default_ctl, dep_bilinear_default_select]
default_bilinear_tex_line = ' & '.join(['Bilinear', '-', '-', '-']+ ['${0:.1f}$'.format(100*x) for x in default_bilinear_tex_line]) + '\\\\'
print(default_bilinear_tex_line)

default_1hid_tex_line = [pos_1hid_default_acc, pos_1hid_default_ctl, pos_1hid_default_select] + [dep_1hid_default_acc, dep_1hid_default_ctl, dep_1hid_default_select]
default_1hid_tex_line = ' & '.join(['MLP-1'] +['${0:.1f}$'.format(100*x) for x in default_1hid_tex_line]) + '\\\\'
print(default_1hid_tex_line)

default_2hid_tex_line = [pos_2hid_default_acc, pos_2hid_default_ctl, pos_2hid_default_select] + [dep_2hid_default_acc, dep_2hid_default_ctl, dep_2hid_default_select]
default_2hid_tex_line = ' & '.join(['MLP-2'] +['${0:.1f}$'.format(100*x) for x in default_2hid_tex_line]) + '\\\\'
print(default_2hid_tex_line)

print('\\bottomrule')
print('\\end{tabular}')
