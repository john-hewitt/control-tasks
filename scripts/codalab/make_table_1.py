import json

# Start of table code
print('\\begin{{tabular}}{{l  c c c | c c c}}')
print('\\toprule')
print('\\bf Probe & \\bf PoS & Ctl & \\bf Select. & \\bf Dep & Ctl & \\bf Select.\\\\')
print('\\midrule')

# Probes with default hyperparams
print('\\multicolumn{{7}}{{c}}{{Probes with ``Default'' Hyperparameters}}')
print('\\vspace{{3pt}}\\\\')

pos_dropout_results = json.load(open('summarize_pos_dropout/results.json'))
dep_dropout_results = json.load(open('summarize_dep_dropout/results.json'))

pos_linear_default_acc = pos_dropout_results["dropout"]["0"]['0hid'][0]
pos_linear_default_ctl = pos_dropout_results["dropout"]["1"]['0hid'][0]
pos_linear_default_select = linear_default_acc - linear_default_ctl
dep_bilinear_default_acc = dep_dropout_results["dropout"]["0"]['bilinear'][0]
dep_bilinear_default_ctl = dep_dropout_results["dropout"]["1"]['bilinear'][0]
dep_bilinear_default_select = bilinear_default_acc - bilinear_default_ctl

pos_1hid_default_acc = pos_dropout_results["dropout"]["0"]['1hid'][0]
pos_1hid_default_ctl = pos_dropout_results["dropout"]["1"]['1hid'][0]
pos_1hid_default_select = 1hid_default_acc - 1hid_default_ctl
dep_1hid_default_acc = dep_dropout_results["dropout"]["0"]['1hid'][0]
dep_1hid_default_ctl = dep_dropout_results["dropout"]["1"]['1hid'][0]
dep_1hid_default_select = 1hid_default_acc - 1hid_default_ctl

pos_2hid_default_acc = pos_dropout_results["dropout"]["0"]['2hid'][0]
pos_2hid_default_ctl = pos_dropout_results["dropout"]["1"]['2hid'][0]
pos_2hid_default_select = 2hid_default_acc - 2hid_default_ctl
dep_2hid_default_acc = dep_dropout_results["dropout"]["0"]['2hid'][0]
dep_2hid_default_ctl = dep_dropout_results["dropout"]["1"]['2hid'][0]
dep_2hid_default_select = 2hid_default_acc - 2hid_default_ctl

default_linear_tex_line = [pos_linear_default_acc, pos_linear_default_ctl, pos_linear_default_select]
default_linear_tex_line = '&'.join(['${.2f}$'.format(x) for x in default_linear_tex_line])
print(default_linear_tex_line)
