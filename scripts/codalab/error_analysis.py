'''
Plots the confusion matrix for part-of-speech tagging
'''

from collections import namedtuple
import itertools
import random
from argparse import ArgumentParser
import json
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
torch.set_printoptions(threshold=5000)
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000
from numpy.ma import masked_array

def generate_lines_for_sent(lines):
  '''Yields batches of lines describing a sentence in conllx.

  Args:
    lines: Each line of a conllx file.
  Yields:
    a list of lines describing a single sentence in conllx.
  '''
  buf = []
  for line in lines:
    if line.startswith('#'):
      continue
    if not line.strip():
      if buf:
        yield buf
        buf = []
      else:
        continue
    else:
      buf.append(line.strip())
  if buf:
    yield buf

Observation = namedtuple('Observation', ['index', 'sentence', 'lemma_sentence', 'upos_sentence', 'xpos_sentence', 'morph', 'head_indices', 'governance_relations', 'secondary_relations', 'extra_info'])

vocab = {'IN': 0, 'DT': 1, 'NNP': 2, 'CD': 3, 'NN': 4, '``': 5, "''": 6, 'POS': 7, '-LRB-': 8, 'VBN': 9, 'NNS': 10, 'VBP': 11, ',': 12, 'CC': 13, '-RRB-': 14, 'VBD': 15, 'RB': 16, 'TO': 17, '.': 18, 'VBZ': 19, 'NNPS': 20, 'PRP': 21, 'PRP$': 22, 'VB': 23, 'JJ': 24, 'MD': 25, 'VBG': 26, 'RBR': 27, ':': 28, 'WP': 29, 'WDT': 30, 'JJR': 31, 'PDT': 32, 'RBS': 33, 'WRB': 34, 'JJS': 35, '$': 36, 'RP': 37, 'FW': 38, 'EX': 39, 'SYM': 40, '#': 41, 'LS': 42, 'UH': 43, 'WP$': 44}
inv_vocab = {i:s for (s,i) in vocab.items()}



def load_observations(conllx_filepath):
    observations = []
    lines = (x for x in open(conllx_filepath))
    for buf in generate_lines_for_sent(lines):
      conllx_lines = []
      for line in buf:
        conllx_lines.append(line.strip().split('\t'))
      embeddings = [None for x in range(len(conllx_lines))]
      observation = Observation(*zip(*conllx_lines))
      observations.append(observation)
    return observations

  



if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('confusion_one_filepath')
  argp.add_argument('confusion_two_filepath')
  args = argp.parse_args()

  confusion_one_matrix = torch.load(args.confusion_one_filepath) 
  confusion_two_matrix = torch.load(args.confusion_two_filepath) 
  keep_count = 45
  
  confusion_matrix = (confusion_one_matrix - confusion_two_matrix)/10
  # Choose gold POS tags for summary
  biggest_diffs = torch.sum(torch.abs(confusion_matrix), 1)
  gold_biggest_diff_indices = torch.argsort(-biggest_diffs)[:keep_count]

  # Choose predicted POS tags for summary
  biggest_diffs = torch.sum(torch.abs(confusion_matrix), 0)
  pred_biggest_diff_indices = torch.argsort(-biggest_diffs)[:keep_count]

  all_biggest_diff_indices = list(sorted(set([int(x) for x in pred_biggest_diff_indices]).union(set([int(x) for x in gold_biggest_diff_indices]))))
  print(all_biggest_diff_indices)
  keep_count = len(all_biggest_diff_indices)

  indices = torch.zeros(45)
  for index in all_biggest_diff_indices:
    indices[index] = 1
  indices = indices.byte()

  #confusion_matrix = (confusion_one_matrix / torch.sum(confusion_one_matrix, 1)) -  (confusion_two_matrix / torch.sum(confusion_two_matrix, 1))
  confusion_matrix = confusion_matrix[:,indices]
  confusion_matrix = confusion_matrix[indices,:]

  fig = plt.figure(figsize=(40,40))
  ax = fig.add_subplot(1,1,1)


  #ax.matshow(confusion_diag,cmap=)
  ax.matshow(torch.abs(confusion_matrix),cmap=mpl.colors.ListedColormap(sns.color_palette("Greys", 256)), vmax=66.8)
  # Set the title of plot
  ax.set_title("Difference of Confusion Matrices, MLP-Linear")
  ax.set_ylabel("Gold Part-of-Speech")
  ax.set_xlabel("Predicted Part-of-Speech")
  ax.grid(False)
  ax.tick_params(axis=u'both', which=u'both',length=0)
  palette = itertools.cycle(sns.color_palette())
  color1 = next(palette)
  color = next(palette)
  color = next(palette)
  color2 = next(palette)
  for i in range(keep_count):
    for j in range(keep_count):
      c = confusion_matrix[j,i]
      color = color1 if (i==j and c>0) or (i!=j and c<0) else color2
      ax.text(i, j, "{0:.2f}".format(c), va='center', ha='center',fontsize=10, color=color, fontweight='bold')


  #ax.set_yticklabels([''] +list(sorted(vocab.keys(), key=lambda x: vocab[x])),fontsize=9)
  ax.set_yticklabels([''] +[inv_vocab[int(x)] for x in all_biggest_diff_indices],fontsize=9)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  #ax.set_xticklabels([''] + list(sorted(vocab.keys(), key=lambda x: vocab[x])),fontsize=9, rotation=90)
  ax.set_xticklabels([''] +[inv_vocab[int(x)] for x in all_biggest_diff_indices],fontsize=9)

  plt.tight_layout()
  plt.savefig('confusion.png', dpi=200)


