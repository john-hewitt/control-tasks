pip install allennlp

allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5  --cuda-device 0 /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.dev.txt /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.dev.elmo-layers.hdf5
allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5  --cuda-device 0 /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.test.txt /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.test.elmo-layers.hdf5
allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5  --cuda-device 0 /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.train.txt /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.train.elmo-layers.hdf5

