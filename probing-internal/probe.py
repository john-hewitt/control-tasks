"""Classes for specifying probe pytorch modules."""

import torch.nn as nn
import torch
import numpy
from tqdm import tqdm

class Probe(nn.Module):

  def print_param_count(self):
    total_params = 0
    for param in self.parameters():
      total_params += numpy.prod(param.size())
    tqdm.write('Probe has {} parameters'.format(total_params))

class TwoWordPSDProbe(Probe):
  """ Computes squared L2 distance after projection by a matrix.

  For a batch of sentences, computes all n^2 pairs of distances
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('Constructing TwoWordPSDProbe')
    super(TwoWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    """ Computes all n^2 pairs of distances after projection
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = transformed.size()
    transformed = transformed.unsqueeze(2)
    transformed = transformed.expand(-1, -1, seqlen, -1)
    transposed = transformed.transpose(1,2)
    diffs = transformed - transposed
    squared_diffs = diffs.pow(2)
    squared_distances = torch.sum(squared_diffs, -1)
    return squared_distances



class OneWordPSDProbe(Probe):
  """ Computes squared L2 norm of words after projection by a matrix."""

  def __init__(self, args):
    print('Constructing OneWordPSDProbe')
    super(OneWordPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (Bh_i)^T(Bh_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = transformed.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        transformed.view(batchlen* seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class OneWordNonPSDProbe(Probe):
  """Computes a bilinear affinity between each word representation and itself.
  
  This is different from the probes in A Structural Probe... as the
  matrix in the quadratic form is not guaranteed positive semi-definite
  
  """

  def __init__(self, args):
    print('Constructing OneWordNonPSDProbe')
    super(OneWordNonPSDProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (h_i^T)A(h_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
    transformed = torch.matmul(batch, self.proj)
    batchlen, seqlen, rank = batch.size()
    norms = torch.bmm(transformed.view(batchlen* seqlen, 1, rank),
        batch.view(batchlen*seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms

class TwoWordNonPSDProbe(Probe):
  """ Computes a bilinear function of difference vectors.

  For a batch of sentences, computes all n^2 pairs of scores
  for each sentence in the batch.
  """
  def __init__(self, args):
    print('TwoWordNonPSDProbe')
    super(TwoWordNonPSDProbe, self).__init__()
    self.args = args
    self.probe_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.model_dim))
    nn.init.uniform_(self.proj, -0.05, 0.05)
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    """ Computes all n^2 pairs of difference scores 
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (h_i-h_j)^TA(h_i-h_j) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    """
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    diffs = (batch_square - batch_square.transpose(1,2)).view(batchlen*seqlen*seqlen, rank)
    psd_transformed = torch.matmul(diffs, self.proj).view(batchlen*seqlen*seqlen,1,rank)
    dists = torch.bmm(psd_transformed, diffs.view(batchlen*seqlen*seqlen, rank, 1))
    dists = dists.view(batchlen, seqlen, seqlen)
    return dists


class OneWordNNDepthProbe(Probe):

  def __init__(self, args):
    print('Constructing OneWordNNDepthProbe')
    super(OneWordNNDepthProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    intermediate_size = 300
    self.initial_linear = nn.Linear(self.model_dim, intermediate_size)
    self.intermediate_linears = nn.ModuleList()
    for i in range(args['probe']['probe_spec']['probe_hidden_layers']):
      self.intermediate_linears.append(nn.Linear(intermediate_size, intermediate_size))
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    batchlen, seqlen, dimension = batch.size()
    batch = self.initial_linear(batch)
    batch = torch.relu(batch)
    for index, linear in enumerate(self.intermediate_linears):
      batch = linear(batch)
      if index != len(self.intermediate_linears) - 1:
        batch = torch.relu(batch)
    batchlen, seqlen, rank = batch.size()
    norms = torch.bmm(batch.view(batchlen* seqlen, 1, rank),
        batch.view(batchlen* seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms


class TwoWordFeaturizedLinearLabelProbe(Probe):
  def __init__(self, args):
    print('Constructing TwoWordFeaturizedLinearLabelProbe')
    super(TwoWordFeaturizedLinearLabelProbe, self).__init__()
    self.args = args
    self.maximum_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.logit_linear_1 = nn.Linear(self.model_dim*4, self.maximum_rank)
    self.logit_linear_2 = nn.Linear(self.maximum_rank, 1)
    self.to(args['device'])
    self.print_param_count()

  def forward(self, batch):
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    batch_transpose = batch_square.transpose(1,2)
    #concat_batch = torch.cat((batch_square, batch_transpose), dim=3)
    diff_batch = batch_square - batch_transpose
    cdot_batch = batch_square * batch_transpose
    concat_batch = torch.cat((batch_square,batch_transpose, diff_batch, cdot_batch),dim=3)
    bottleneck = self.logit_linear_1(concat_batch) 
    logits = self.logit_linear_2(bottleneck).view(batchlen, seqlen, seqlen)
    return logits

class TwoWordLinearLabelProbe(Probe):
  def __init__(self, args):
    print('Constructing TwoWordLinearLabelProbe')
    super(TwoWordLinearLabelProbe, self).__init__()
    self.args = args
    self.maximum_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.logit_linear_1 = nn.Linear(self.model_dim*2, self.maximum_rank)
    self.logit_linear_2 = nn.Linear(self.maximum_rank, 1)
    self.to(args['device'])
    self.print_param_count()
    self.dropout = nn.Dropout(p=args['probe']['dropout'])
    print('Applying dropout {}'.format(args['probe']['dropout']))
    print('Using intermediate size (hidden dim / rank) {}'.format(self.maximum_rank))

  def forward(self, batch):
    batch = self.dropout(batch)
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    concat_batch = torch.cat((batch_square, batch_square.transpose(1,2)), dim=3)
    bottleneck = self.logit_linear_1(concat_batch) 
    logits = self.logit_linear_2(bottleneck).view(batchlen, seqlen, seqlen)
    return logits

class TwoWordBilinearLabelProbe(Probe):
  def __init__(self, args):
    print('Constructing TwoWordBilinearLabelProbe')
    super(TwoWordBilinearLabelProbe, self).__init__()
    self.args = args
    self.maximum_rank = args['probe']['maximum_rank']
    self.model_dim = args['model']['hidden_dim']
    self.proj_L = nn.Parameter(data = torch.zeros(self.model_dim, self.maximum_rank))
    self.proj_R = nn.Parameter(data = torch.zeros(self.maximum_rank, self.model_dim))
    self.bias = nn.Parameter(data=torch.zeros(1))
    nn.init.uniform_(self.proj_L, -0.05, 0.05)
    nn.init.uniform_(self.proj_R, -0.05, 0.05)
    nn.init.uniform_(self.bias, -0.05, 0.05)
    self.to(args['device'])
    self.print_param_count()
    self.dropout = nn.Dropout(p=args['probe']['dropout'])
    print('Applying dropout {}'.format(args['probe']['dropout']))
    print('Using intermediate size (hidden dim / rank) {}'.format(self.maximum_rank))

  def forward(self, batch):
    batchlen, seqlen, rank = batch.size()
    batch = self.dropout(batch)
    proj = torch.mm(self.proj_L, self.proj_R)
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    batch_transposed = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank).contiguous().view(batchlen*seqlen*seqlen,rank,1)

    psd_transformed = torch.matmul(batch_square.contiguous(), proj).view(batchlen*seqlen*seqlen,1, rank)
    logits = (torch.bmm(psd_transformed, batch_transposed) + self.bias).view(batchlen, seqlen, seqlen)
    return logits

class TwoWordNNLabelProbe(Probe):
  def __init__(self, args):
    print('Constructing TwoWordNNLabelProbe')
    super(TwoWordNNLabelProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.maximum_rank = args['probe']['maximum_rank']
    self.hidden_layers = args['probe']['probe_spec']['probe_hidden_layers']
    if self.hidden_layers == 2:
      self.linearfirst = nn.Linear(self.model_dim*2, self.maximum_rank)
      self.linearmid = nn.Linear(self.maximum_rank, self.maximum_rank)
      self.linearlast = nn.Linear(self.maximum_rank, 1)
    else:
      self.linearfirst = nn.Linear(self.model_dim*2, self.maximum_rank)
      self.linearlast = nn.Linear(self.maximum_rank, 1)
    self.to(args['device'])
    self.print_param_count()
    self.dropout = nn.Dropout(p=args['probe']['dropout'])
    print('Applying dropout {}'.format(args['probe']['dropout']))
    print('Using intermediate size (hidden dim / rank) {}'.format(self.maximum_rank))

  def _forward_1(self, batch):
    batchlen, seqlen, rank = batch.size()
    batch = self.dropout(batch)
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    batch_square_transpose = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank)
    concat_batch = torch.cat((batch_square, batch_square_transpose), dim=3)
    hidden = torch.relu(self.linearfirst(concat_batch))
    hidden = self.dropout(hidden)
    logits = self.linearlast(hidden).view(batchlen,seqlen,seqlen)
    return logits

  def _forward_2(self, batch):
    batchlen, seqlen, rank = batch.size()
    batch = self.dropout(batch)
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    batch_square_transpose = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank)
    concat_batch = torch.cat((batch_square, batch_square_transpose), dim=3)
    hidden1 = torch.relu(self.linearfirst(concat_batch))
    hidden1 = self.dropout(hidden1)
    hidden2 = torch.relu(self.linearmid(hidden1))
    hidden2 = self.dropout(hidden2)
    logits = self.linearlast(hidden2).view(batchlen,seqlen,seqlen)
    return logits

  def forward(self, batch):
    if self.hidden_layers == 2:
      return self._forward_2(batch)
    else:
      return self._forward_1(batch)


class OneWordLinearLabelProbe(Probe):

  def __init__(self, args):
    print('Constructing OneWordLinearLabelProbe')
    super(OneWordLinearLabelProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.label_space_size = args['probe']['label_space_size']
    self.maximum_rank = args['probe']['maximum_rank']
    self.linear1 = nn.Linear(self.model_dim, self.maximum_rank)
    self.linear2 = nn.Linear(self.maximum_rank, self.label_space_size)
    self.to(args['device'])
    self.print_param_count()
    self.dropout = nn.Dropout(p=args['probe']['dropout'])
    print('Applying dropout {}'.format(args['probe']['dropout']))
    print('Using intermediate size (hidden dim / rank) {}'.format(self.maximum_rank))

  def forward(self, batch):
    batchlen, seqlen, dimension = batch.size()
    batch = self.dropout(batch)
    batch = self.linear1(batch)
    logits = self.linear2(batch)
    return logits

class OneWordNNLabelProbe(Probe):

  def __init__(self, args):
    print('Constructing OneWordNNLabelProbe')
    super(OneWordNNLabelProbe, self).__init__()
    self.args = args
    self.model_dim = args['model']['hidden_dim']
    self.label_space_size = args['probe']['label_space_size']
    intermediate_size = args['probe']['maximum_rank']
    self.initial_linear = nn.Linear(self.model_dim, intermediate_size)
    self.intermediate_linears = nn.ModuleList()
    for i in range(args['probe']['probe_spec']['probe_hidden_layers']-1):
      self.intermediate_linears.append(nn.Linear(intermediate_size, intermediate_size))
    self.last_linear = nn.Linear(intermediate_size, self.label_space_size)
    self.to(args['device'])
    self.print_param_count()
    self.dropout = nn.Dropout(p=args['probe']['dropout'])
    print('Applying dropout {}'.format(args['probe']['dropout']))
    print('Using intermediate size (hidden dim / rank) {}'.format(intermediate_size))

  def forward(self, batch):
    batchlen, seqlen, dimension = batch.size()
    batch = self.dropout(batch)
    batch = self.initial_linear(batch)
    batch = torch.relu(batch)
    for linear in self.intermediate_linears:
      batch = linear(batch)
      batch = torch.relu(batch)
      batch = self.dropout(batch)
    batch = self.last_linear(batch)
    return batch
