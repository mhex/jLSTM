# jLSTM
Multithreaded (asynchronous SGD) implementation of protein/gene sequence classification with LSTM.
This was ported 2009/2010 from the single threaded C sources which we used to do the experiments
in https://academic.oup.com/bioinformatics/article/23/14/1728/189356/Fast-model-based-protein-homology-detection

Each thread randomly picks a sequence, copies the global weight matrix in its thread space and does
forward- and backpropagation. Therefore, forward- and backpropagation only rely on the local weight matrix.
After a write lock on the global matrix the deltas are applied and the write lock is released. This is repeated for
every thread in parallel until a stopping criterion is reached. Every thread maintains its own shuffeled sequence list.
Once the sequence list is trained by a thread a global epoch counter is incremented secured by a write lock.
After a given number of epochs the first thread reaching this number is doing a forward propagation on some held out test sequences by grabbing the current weight matrix. The other threads continue training. Training and test metrics are accuracy,
number of false and true positives and negatives as well the AUC of the ROC and ROC50.

see also

http://bioinf.jku.at/software/LSTM_protein/
