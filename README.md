##### Common swedish name classification

This code is the implementation of a recurrent neural net in pytorch. The implementation is for classifying common swedish names into gender categories. It is a character level rnn, ant the net iteself is a bi-directional 2-layer lstm. It also has batch-feeding in the training part, whih makes it faster. Variable batch size training makes it versatile as well.

Also availible as kernel at https://www.kaggle.com/geeklund/pytorch-rnn-text-classification/notebook