'''
Creates the accuracy-across-rank plot for 

A Structural Probe for Finding Syntax in Word Representations
NAACL '19

'''

from collections import defaultdict
import itertools
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from matplotlib.ticker import FormatStrFormatter


palette = itertools.cycle(sns.color_palette())
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)
color =next(palette)

elmo_color = color
bertlarge_color = next(palette)
bertlarge_color = next(palette)
bertlarge_color = next(palette)
bertlarge_color = next(palette)
bertlarge_color = next(palette)
bertlarge_color = next(palette)

bertbase_color = next(palette)

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#ranks = [2**j for j in range(10)]
ranks = [j for j in range(10)]
elmo1_uuas = []

# FOR POS expts

# Sample
#sample_x_axis = [400, 4000, 40000]
#sample_c0_elmo1_0hid = [0.955156168207991, 0.967245806017399, 0.971957025699828]
#sample_c0_elmo1_1hid = [0.955604855796794, 0.968766358401675, 0.9744746616147768]
#sample_c0_elmo1_2hid = [0.953909813794650, 0.967071316399531, 0.9736520677019718]
#sample_c1_elmo1_0hid = [0.610115412418675, 0.695989231497868, 0.7165042251414612]
#sample_c1_elmo1_1hid = [0.643717127402348, 0.83602961338086, 0.9279607149088915]
#sample_c1_elmo1_2hid = [0.625545280055836, 0.823740558865318, 0.9205075155171124]

sample_result_dict = json.load(open('summarize_pos_sample/results.json'))
sample_x_axis = sample_result_dict["hyperparameter_options"]
sample_c1_elmo1_0hid = sample_result_dict["sample"]["1"]["0hid"]
sample_c1_elmo1_1hid = sample_result_dict["sample"]["1"]["1hid"]
sample_c1_elmo1_2hid = sample_result_dict["sample"]["1"]["2hid"]
sample_c0_elmo1_0hid = sample_result_dict["sample"]["0"]["0hid"]
sample_c0_elmo1_1hid = sample_result_dict["sample"]["0"]["1hid"]
sample_c0_elmo1_2hid = sample_result_dict["sample"]["0"]["2hid"]

sample_diff_elmo1_1hid = [x-y for x, y in zip(sample_c0_elmo1_1hid, sample_c1_elmo1_1hid)]
sample_diff_elmo1_2hid = [x-y for x, y in zip(sample_c0_elmo1_2hid, sample_c1_elmo1_2hid)]
sample_diff_elmo1_0hid = [x-y for x, y in zip(sample_c0_elmo1_0hid, sample_c1_elmo1_0hid)]

# Rank
#rank_x_axis = [2, 4, 10, 45, 1000]
#rank_c0_elmo1_0hid = [0.852381783283894, 0.952364334322107, 0.969564025226213, 0.97113443178702, 0.971957025699828]
#rank_c0_elmo1_1hid = [0.835730488321659, 0.94501084328339, 0.970561108756886, 0.972355859112097, 0.9742503178203754]
#rank_c0_elmo1_2hid = [0.81337089014632, 0.929207069322232, 0.968866066754742, 0.973851484408106, 0.9741007552907744]
#rank_c1_elmo1_0hid = [0.403744048657676, 0.533040855497669, 0.646135054964229, 0.716379589700127, 0.7165042251414612]
#rank_c1_elmo1_1hid = [0.392925692349876, 0.556572026821546, 0.674103247999601, 0.805419148989206, 0.9278610065558243]
#rank_c1_elmo1_2hid = [0.381833138071141, 0.523144801455741, 0.672931674851060, 0.816736047062342, 0.9223271929605903]
#rank_x_axis = [2.0, 4.0, 10.0, 45.0, 1000.0]
#rank_c0_elmo1_0hid = [0.8367774260288656, 0.9516414487623701, 0.9700127128150161, 0.9726300570830322, 0.9716329735523593]
#rank_c1_elmo1_0hid = [0.41832639529376575, 0.5413415758905202, 0.6401276266919261, 0.7120173492534337, 0.7119924221651669]
#rank_c0_elmo1_1hid = [0.7709948400927288, 0.9491736670239549, 0.9694892439614129, 0.9724804945534312, 0.9737767031433058]
#rank_c1_elmo1_1hid = [0.4190742079417703, 0.5349103871176808, 0.668195528080365, 0.8063165241668121, 0.9287334546451629]
#rank_c0_elmo1_2hid = [0.8018795024553182, 0.9480768751402149, 0.9699379315502156, 0.9718573173467607, 0.9732781613779694]
#rank_c1_elmo1_2hid = [0.4130169254929332, 0.5547024952015355, 0.6662761422838198, 0.8174090784455468, 0.9228008076376598]

rank_result_dict = json.load(open('summarize_pos_rank/results.json'))
rank_x_axis = rank_result_dict["hyperparameter_options"]
rank_c1_elmo1_0hid = rank_result_dict["rank"]["1"]["0hid"]
rank_c1_elmo1_1hid = rank_result_dict["rank"]["1"]["1hid"]
rank_c1_elmo1_2hid = rank_result_dict["rank"]["1"]["2hid"]
rank_c0_elmo1_0hid = rank_result_dict["rank"]["0"]["0hid"]
rank_c0_elmo1_1hid = rank_result_dict["rank"]["0"]["1hid"]
rank_c0_elmo1_2hid = rank_result_dict["rank"]["0"]["2hid"]

rank_diff_elmo1_1hid = [x-y for x, y in zip(rank_c0_elmo1_1hid, rank_c1_elmo1_1hid)]
rank_diff_elmo1_2hid = [x-y for x, y in zip(rank_c0_elmo1_2hid, rank_c1_elmo1_2hid)]
rank_diff_elmo1_0hid = [x-y for x, y in zip(rank_c0_elmo1_0hid, rank_c1_elmo1_0hid)]

# Dropout
#dropout_x_axis = list(reversed([0, 0.2, 0.4, 0.6, 0.8]))
#dropout_c0_elmo1_0hid = list(reversed([0.97195702569982, 0.972056734052895, 0.971608046464092, 0.96891592093127, 0.9626093675997707]))
#dropout_c0_elmo1_1hid = list(reversed([0.974474661614776, 0.974424807438243, 0.973951192761173, 0.974001046937707, 0.9705112545803525]))
#dropout_c0_elmo1_2hid = list(reversed([0.973652067701971, 0.973452650995837, 0.973303088466236, 0.973976119849440, 0.9705361816686193]))
#dropout_c1_elmo1_0hid = list(reversed([0.716504225141461, 0.70369170177231, 0.680434728419373, 0.646683450906099, 0.5812498442056984]))
#dropout_c1_elmo1_1hid = list(reversed([0.927960714908891, 0.935787820624672, 0.931699778148914, 0.914724431039210, 0.8651444524765062]))
#dropout_c1_elmo1_2hid = list(reversed([0.920507515517112, 0.940848019542837, 0.939377321335094, 0.909689159209312, 0.8348331131440536]))

dropout_result_dict = json.load(open('summarize_pos_dropout/results.json'))
dropout_x_axis = dropout_result_dict["hyperparameter_options"]
dropout_c1_elmo1_0hid = dropout_result_dict["dropout"]["1"]["0hid"]
dropout_c1_elmo1_1hid = dropout_result_dict["dropout"]["1"]["1hid"]
dropout_c1_elmo1_2hid = dropout_result_dict["dropout"]["1"]["2hid"]
dropout_c0_elmo1_0hid = dropout_result_dict["dropout"]["0"]["0hid"]
dropout_c0_elmo1_1hid = dropout_result_dict["dropout"]["0"]["1hid"]
dropout_c0_elmo1_2hid = dropout_result_dict["dropout"]["0"]["2hid"]

dropout_diff_elmo1_1hid = [x-y for x, y in zip(dropout_c0_elmo1_1hid, dropout_c1_elmo1_1hid)]
dropout_diff_elmo1_2hid = [x-y for x, y in zip(dropout_c0_elmo1_2hid, dropout_c1_elmo1_2hid)]
dropout_diff_elmo1_0hid = [x-y for x, y in zip(dropout_c0_elmo1_0hid, dropout_c1_elmo1_0hid)]

# Weight Decay
#weightdecay_x_axis = list(reversed([0, 0.01, 0.1, 1.0]))
#weightdecay_c0_elmo1_0hid = list(reversed([0.97195702569982, 0.971458483934491, 0.962060971657900, 0.8603335244410101]))
#weightdecay_c0_elmo1_1hid = list(reversed([0.974474661614776, 0.972679911259565, 0.962185607099234, 0.8555724505820476]))
#weightdecay_c0_elmo1_2hid = list(reversed([0.973652067701971, 0.973278161377969, 0.963880649101378, 0.8263329760450682]))
#weightdecay_c1_elmo1_0hid = list(reversed([0.716504225141461, 0.713114141137173, 0.666525413166488, 0.4413340977640402]))
#weightdecay_c1_elmo1_1hid = list(reversed([0.927960714908891, 0.909090909090909, 0.737368198020789, 0.439290076526161]))
#weightdecay_c1_elmo1_2hid = list(reversed([0.920507515517112, 0.928234912879826, 0.810155295759902, 0.3618416132811526]))

wd_result_dict = json.load(open('summarize_pos_wd/results.json'))
weightdecay_x_axis = wd_result_dict["hyperparameter_options"]
weightdecay_c1_elmo1_0hid = wd_result_dict["wd"]["1"]["0hid"]
weightdecay_c1_elmo1_1hid = wd_result_dict["wd"]["1"]["1hid"]
weightdecay_c1_elmo1_2hid = wd_result_dict["wd"]["1"]["2hid"]
weightdecay_c0_elmo1_0hid = wd_result_dict["wd"]["0"]["0hid"]
weightdecay_c0_elmo1_1hid = wd_result_dict["wd"]["0"]["1hid"]
weightdecay_c0_elmo1_2hid = wd_result_dict["wd"]["0"]["2hid"]


weightdecay_diff_elmo1_1hid = [x-y for x, y in zip(weightdecay_c0_elmo1_1hid, weightdecay_c1_elmo1_1hid)]
weightdecay_diff_elmo1_2hid = [x-y for x, y in zip(weightdecay_c0_elmo1_2hid, weightdecay_c1_elmo1_2hid)]
weightdecay_diff_elmo1_0hid = [x-y for x, y in zip(weightdecay_c0_elmo1_0hid, weightdecay_c1_elmo1_0hid)]

# Gradient Steps
#gradsteps_x_axis = [1500, 3000, 6000, 12500, 25000, 50000]
#gradsteps_c0_elmo1_0hid = [0.970536181668619, 0.971832390258493, 0.972081661141162, 0.972281077847296, 0.970885160904354, 0.9711593588752898]
#gradsteps_c0_elmo1_1hid = [0.97180746317022, 0.972879327965700, 0.974225390732108, 0.974723932497444, 0.973951192761173, 0.9735523593489045]
#gradsteps_c0_elmo1_2hid = [0.973078744671834, 0.97484856793877, 0.974474661614776, 0.973303088466236, 0.973103671760101, 0.9714086297579579]
#gradsteps_c1_elmo1_0hid = [0.697983398559214, 0.707405837924072, 0.713737318343844, 0.713488047461176, 0.715158162375052, 0.716254954258793]
#gradsteps_c1_elmo1_1hid = [0.857591544731659, 0.894010020689483, 0.913478076625869, 0.927487100231821, 0.93065284044170, 0.9286337462920956]
#gradsteps_c1_elmo1_2hid = [0.875339631577635, 0.896577510780965, 0.911608545005857, 0.93244759079691, 0.931400653089712, 0.9225764638432584]

gradient_steps_result_dict = json.load(open('summarize_pos_gradient_steps/results.json'))
gradsteps_x_axis = gradient_steps_result_dict["hyperparameter_options"]
gradsteps_c1_elmo1_0hid = gradient_steps_result_dict["gradient_steps"]["1"]["0hid"]
gradsteps_c1_elmo1_1hid = gradient_steps_result_dict["gradient_steps"]["1"]["1hid"]
gradsteps_c1_elmo1_2hid = gradient_steps_result_dict["gradient_steps"]["1"]["2hid"]
gradsteps_c0_elmo1_0hid = gradient_steps_result_dict["gradient_steps"]["0"]["0hid"]
gradsteps_c0_elmo1_1hid = gradient_steps_result_dict["gradient_steps"]["0"]["1hid"]
gradsteps_c0_elmo1_2hid = gradient_steps_result_dict["gradient_steps"]["0"]["2hid"]

gradsteps_diff_elmo1_1hid = [x-y for x, y in zip(gradsteps_c0_elmo1_1hid, gradsteps_c1_elmo1_1hid)]
gradsteps_diff_elmo1_2hid = [x-y for x, y in zip(gradsteps_c0_elmo1_2hid, gradsteps_c1_elmo1_2hid)]
gradsteps_diff_elmo1_0hid = [x-y for x, y in zip(gradsteps_c0_elmo1_0hid, gradsteps_c1_elmo1_0hid)]

fig = plt.figure(figsize=(10,3.5))
#fig = plt.figure(figsize=(10,4.5))
#fig.suptitle('Regularization / Model Constraint Method')
fig.suptitle(r'\bf Part-of-speech Accuracy and Selectivity Across Complexity Control Methods')
#plt.subplots_adjust(top=.3)
ax1 = plt.subplot(2,5,1)
ax1.set_title('Sample Count')
sns.despine(offset=6)#, trim=True);
ax1.plot([str(x) for x in sample_x_axis], sample_c0_elmo1_2hid, markersize=4, marker='p')
ax1.plot([str(x) for x in sample_x_axis], sample_c0_elmo1_1hid, markersize=4, marker='+')
ax1.plot([str(x) for x in sample_x_axis], sample_c0_elmo1_0hid, markersize=4, marker='s')
ax1.set_ylim((.94,.98))
#ax1.plot([str(x) for x in sample_x_axis], [.972 for x in sample_x_axis], linewidth=4)
#ax1.fill_between([str(x) for x in sample_x_axis], np.array([.974 for x in sample_x_axis]) -.01, np.array([.974 for x in sample_x_axis]), alpha=.5)

ax2 = plt.subplot(2,5,6)
sns.despine(offset=6)#, trim=True);
ax2.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_2hid, markersize=4, marker='p')
ax2.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_1hid, markersize=4, marker='+')
ax2.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_0hid, markersize=4, marker='s')
#ax2.fill_between([str(x) for x in sample_x_axis], np.array([.24 for x in sample_x_axis]) -.05, np.array([.24 for x in sample_x_axis]) +.05, alpha=.5)

#ax11 = plt.subplot(3,5,11)
#sns.despine(offset=6)#, trim=True);
#ax11.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_2hid, markersize=4, marker='p')
#ax11.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_1hid, markersize=4, marker='+')
#ax11.plot([str(x) for x in sample_x_axis], sample_diff_elmo1_0hid, markersize=4, marker='s')

ax3 = plt.subplot(2,5,2, sharey=ax1)
ax3.set_title('Rank/Hidden Dim')
sns.despine(offset=6)#, trim=True);
ax3.plot([str(x) for x in rank_x_axis], rank_c0_elmo1_2hid, markersize=4, marker='p')
ax3.plot([str(x) for x in rank_x_axis], rank_c0_elmo1_1hid, markersize=4, marker='+')
ax3.plot([str(x) for x in rank_x_axis], rank_c0_elmo1_0hid, markersize=4, marker='s')
#ax3.fill_between([str(x) for x in rank_x_axis], np.array([.974 for x in rank_x_axis]) -.01, np.array([.974 for x in rank_x_axis]), alpha=.5)
ax3.tick_params(labelleft=False)

ax4 = plt.subplot(2,5,7, sharey=ax2)
sns.despine(offset=6)#, trim=True);
ax4.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_2hid, markersize=4, marker='p')
ax4.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_1hid, markersize=4, marker='+')
ax4.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_0hid, markersize=4, marker='s')
#ax4.fill_between([str(x) for x in rank_x_axis], np.array([.24 for x in rank_x_axis]) -.05, np.array([.24 for x in rank_x_axis]) +.05, alpha=.5)
ax4.tick_params(labelleft=False)

#ax12 = plt.subplot(3,5,12)
#sns.despine(offset=6)#, trim=True);
#ax12.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_2hid, markersize=4, marker='p')
#ax12.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_1hid, markersize=4, marker='+')
#ax12.plot([str(x) for x in rank_x_axis], rank_diff_elmo1_0hid, markersize=4, marker='s')


ax5 = plt.subplot(2,5,3, sharey=ax1)
ax5.set_title('Dropout')
sns.despine(offset=6)#, trim=True);
ax5.plot([str(x) for x in dropout_x_axis], dropout_c0_elmo1_2hid, markersize=4, marker='p')
ax5.plot([str(x) for x in dropout_x_axis], dropout_c0_elmo1_1hid, markersize=4, marker='+')
ax5.plot([str(x) for x in dropout_x_axis], dropout_c0_elmo1_0hid, markersize=4, marker='s')
#ax5.fill_between([str(x) for x in dropout_x_axis], np.array([.974 for x in dropout_x_axis]) -.01, np.array([.974 for x in dropout_x_axis]), alpha=.5)
ax5.tick_params(labelleft=False)

ax6 = plt.subplot(2,5,8, sharey=ax2)
sns.despine(offset=6)#, trim=True);
ax6.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_2hid, markersize=4, marker='p')
ax6.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_1hid, markersize=4, marker='+')
ax6.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_0hid, markersize=4, marker='s')
#ax6.fill_between([str(x) for x in dropout_x_axis], np.array([.24 for x in dropout_x_axis]) -.05, np.array([.24 for x in dropout_x_axis]) +.05, alpha=.5)
ax6.tick_params(labelleft=False)

#ax13 = plt.subplot(3,5,13)
#sns.despine(offset=6)#, trim=True);
#ax13.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_2hid, markersize=4, marker='p')
#ax13.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_1hid, markersize=4, marker='+')
#ax13.plot([str(x) for x in dropout_x_axis], dropout_diff_elmo1_0hid, markersize=4, marker='s')

ax7 = plt.subplot(2,5,4, sharey=ax1)
sns.despine(offset=6)#, trim=True);
ax7.set_title('Weight Decay')
ax7.plot([str(x) for x in weightdecay_x_axis], weightdecay_c0_elmo1_2hid, markersize=4, marker='p')
ax7.plot([str(x) for x in weightdecay_x_axis], weightdecay_c0_elmo1_1hid, markersize=4, marker='+')
ax7.plot([str(x) for x in weightdecay_x_axis], weightdecay_c0_elmo1_0hid, markersize=4, marker='s')
#ax7.fill_between([str(x) for x in weightdecay_x_axis], np.array([.974 for x in weightdecay_x_axis]) -.01, np.array([.974 for x in weightdecay_x_axis]), alpha=.5)
ax7.tick_params(labelleft=False)

ax8 = plt.subplot(2,5,9, sharey=ax2)
sns.despine(offset=6)#, trim=True);
ax8.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_2hid, markersize=4, marker='p')
ax8.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_1hid, markersize=4, marker='+')
ax8.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_0hid, markersize=4, marker='s')
#ax8.fill_between([str(x) for x in weightdecay_x_axis], np.array([.24 for x in weightdecay_x_axis]) -.05, np.array([.24 for x in weightdecay_x_axis]) +.05, alpha=.5)
ax8.tick_params(labelleft=False)

#ax14 = plt.subplot(3,5,14)
#sns.despine(offset=6)#, trim=True);
#ax14.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_2hid, markersize=4, marker='p')
#ax14.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_1hid, markersize=4, marker='+')
#ax14.plot([str(x) for x in weightdecay_x_axis], weightdecay_diff_elmo1_0hid, markersize=4, marker='s')

ax9 = plt.subplot(2,5,5, sharey=ax1)
sns.despine(offset=6)#, trim=True);
ax9.set_title('Gradient Steps')
ax9.plot([str(x) for x in gradsteps_x_axis], gradsteps_c0_elmo1_2hid, markersize=4, marker='p', label='MLP-2')
ax9.plot([str(x) for x in gradsteps_x_axis], gradsteps_c0_elmo1_1hid, markersize=4, marker='+', label='MLP-1')
ax9.plot([str(x) for x in gradsteps_x_axis], gradsteps_c0_elmo1_0hid, markersize=4, marker='s', label='Linear')
#ax9.fill_between([str(x) for x in gradsteps_x_axis], np.array([.974 for x in gradsteps_x_axis]) -.01, np.array([.974 for x in gradsteps_x_axis]), alpha=.5)
ax9.tick_params(labelleft=False)

ax10 = plt.subplot(2,5,10, sharey=ax2)
sns.despine(offset=6)#, trim=True);
ax10.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_2hid, markersize=4, marker='p')
ax10.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_1hid, markersize=4, marker='+')
ax10.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_0hid, markersize=4, marker='s')
ax10.tick_params(axis='x', labelsize=8)
#ax10.fill_between([str(x) for x in gradsteps_x_axis], np.array([.24 for x in gradsteps_x_axis]) -.05, np.array([.24 for x in gradsteps_x_axis]) +.05, alpha=.5)
ax10.tick_params(labelleft=False)

#ax15 = plt.subplot(3,5,15)
#sns.despine(offset=6)#, trim=True);
#ax15.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_2hid, markersize=4, marker='p')
#ax15.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_1hid, markersize=4, marker='+')
#ax15.plot([str(x) for x in gradsteps_x_axis], gradsteps_diff_elmo1_0hid, markersize=4, marker='s')

fig.autofmt_xdate()


ax1.set_ylabel(r'\bf PoS Accuracy')
ax2.set_ylabel(r'\bf Selectivity')

plt.tight_layout()
plt.subplots_adjust(top=0.82)
ax9.legend(fontsize=9)

#ax1.bar(sample_x_axis, sample_c0_elmo1_2hid, width=10000)
#ax1.bar(sample_x_axis, sample_c0_elmo1_0hid, width=10000)

#ax1.set_xticks(sample_x_axis, [str(x) for x in sample_x_axis])
ax1.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax3.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax4.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax5.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax6.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax7.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax8.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax8.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax9.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax9.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax10.yaxis.set_major_locator(plt.LinearLocator(numticks=3))
#ax10.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.savefig('poscontrol', dpi=400)

#bertbase_spearmanr = [x * 100 for x in [0.55, 0.63, 0.71, 0.77, 0.81, 0.83, 0.84, 0.85, 0.85, 0.85]]
#bertbase_uuas = [x * 100 for x in [0.266, 0.389, 0.509, 0.659, 0.742, 0.780, 0.795, 0.799, 0.797, 0.797]]

#plt.figure(figsize=(7,4))
#plt.rc('text', usetex=True)
##plt.rc('font', family='serif')
#plt.plot(ranks, bertbase_uuas, label=r'\textsc{BERTbase7} UUAS', linestyle='solid', color=bertbase_color, marker='o')#, width=.8)
#plt.plot(ranks, bertlarge_uuas, label=r'\textsc{BERTlarge16} UUAS', linestyle='solid', color=bertlarge_color, marker='s')#, width=.8)
#plt.plot(ranks, elmo_uuas, label=r'\textsc{ELMo1} UUAS', linestyle='solid', color=elmo_color, marker='^')#, width=.8)
#
#plt.plot(ranks, bertbase_spearmanr, label=r'\textsc{BERTbase7} DSpr.', linestyle='dotted', color=bertbase_color, marker='o')#, width=.8)
#plt.plot(ranks, bertlarge_spearmanr, label=r'\textsc{BERTlarge16} DSpr.', linestyle='dotted', color=bertlarge_color, marker='s')#, width=.8)
#plt.plot(ranks, elmo_spearmanr, label=r'\textsc{ELMo1} DSpr.', linestyle='dotted', color=elmo_color, marker='^')#, width=.8)
#
##plt.plot(ranks, bertbase_spearmanr)#, width=.8)
#locs, labels = plt.xticks()
#print([x.get_text() for x in labels])
#plt.xticks(ranks, [str(2**j) for j in range(10)])
#plt.tick_params(labelsize=17)
#plt.legend(fontsize=14)
#plt.ylabel('UUAS', fontsize=20)
#plt.xlabel('Probe Maximum Rank',fontsize=20)
#sns.despine(offset=10, trim=True);
#plt.tight_layout()
#plt.savefig('ranks', dpi=300)
#
## Second one...
#plt.clf()
##plt.figure(figsize=(7,3.5))
#fig, ax1 = plt.subplots(figsize=(7,3.5))
#plt.rc('text', usetex=True)
#ax1.plot(ranks, [x*100 for x in bertbase_uuas], label=r'\textsc{BERTbase7} UUAS', linestyle='solid', color=bertbase_color, marker='o')#, width=.8)
#ax1.plot(ranks, [x*100 for x in bertlarge_uuas], label=r'\textsc{BERTlarge16} UUAS', linestyle='solid', color=bertlarge_color, marker='s')#, width=.8)
#ax1.plot(ranks, [x*100 for x in elmo_uuas], label=r'\textsc{ELMo1} UUAS', linestyle='solid', color=elmo_color, marker='^')#, width=.8)
#ax1.set_ylabel('UUAS', fontsize=20)
#
#ax2 = ax1.twinx()
#ax2.plot(ranks, bertbase_spearmanr, label=r'\textsc{BERTbase7} DSpr.', linestyle='dotted', color=bertbase_color, marker='o')#, width=.8)
#ax2.plot(ranks, bertlarge_spearmanr, label=r'\textsc{BERTlarge16} DSpr.', linestyle='dotted', color=bertlarge_color, marker='s')#, width=.8)
#ax2.plot(ranks, elmo_spearmanr, label=r'\textsc{ELMo1} DSpr.', linestyle='dotted', color=elmo_color, marker='^')#, width=.8)
#ax2.set_ylabel('DSpr.', fontsize=20)
#ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#
##plt.plot(ranks, bertbase_spearmanr)#, width=.8)
#locs, labels = plt.xticks()
#print([x.get_text() for x in labels])
#plt.xticks(ranks, [str(2**j) for j in range(10)])
#ax1.tick_params(labelsize=16)
#ax2.tick_params(labelsize=16)
#ax1.legend(fontsize=12, loc=(.47,0))
#ax2.legend(fontsize=12, loc=(.47,0.335))
##plt.ylabel('UUAS', fontsize=20)
#ax1.set_xlabel('Probe Maximum Rank',fontsize=18)
#sns.despine(offset=10, right=False, trim=True);
#plt.tight_layout()
#plt.savefig('ranks2', dpi=300)

