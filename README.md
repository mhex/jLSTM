# jLSTM
Multithreaded (asynchronous SGD) implementation of protein/gene sequence classification with LSTM.
This was ported 2009/2010 from the single threaded C sources which we used to do the experiments
in https://academic.oup.com/bioinformatics/article/23/14/1728/189356/Fast-model-based-protein-homology-detection

Each thread randomly picks a sequence, copies the global weight matrix in its thread space and does
forward- and backpropagation. Therefore, forward- and backpropagation only rely on the local weight matrix.
After a write lock on the global matrix the deltas are applied and the write lock is released. This is repeated for
every thread until a stopping criterion is reached. Occasional forward propagations of some held out test sequences
done by some thread reaching a global step number report the generalization error parallel to the training procedure.

see also

http://bioinf.jku.at/software/LSTM_protein/
