# Downloads example corpora and vectors for structural probing.
# Includes conllx files, raw text files, and ELMo contextual word representations

# By default, downloads a (very) small subset of the EN-EWT
# universal dependencies corpus. Uncomment the full download
# lines below to get a corpus large enough for reasonable results
wget https://nlp.stanford.edu/~johnhew/public/en_ewt-ud-sample.tgz
tar xzvf en_ewt-ud-sample.tgz
mkdir -p example/data
mv en_ewt-ud-sample example/data
rm en_ewt-ud-sample.tgz
