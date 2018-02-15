import mxnet as mx
import numpy as np

import src.read.read_train_example_saved as rtes


def ml_acc(label, pred, label_width=2):
        return float((label == np.round(pred)).sum()) / label_width / pred.shape[0]


def do_model(batch_size=4, nb_hidden_layer=4, data_file_path="../../out/train_example.csv"):

    data_file_path = data_file_path
    pixels = rtes.read_train_example_saved(data_file_path)


    # TODO randomize the selection and create validation set
    ntrain = int(pixels.shape[0]*0.8)

    train = pixels[:ntrain, 3:]
    test = pixels[ntrain:, 3:]
    # print(train.shape, train[0, 0:4])

    labels = pixels[:, 1]
    unique_labels = np.unique(labels)
    label_train = labels[:ntrain]
    label_test = labels[ntrain:]

    for labels_ in unique_labels:

            label_train_one_vs_all = (label_train == labels_)
            label_test_one_vs_all = (label_test == labels_)
            print("#(class = %s): Train: %i - Test: %i" % (labels_, sum(label_train_one_vs_all), sum(label_test_one_vs_all)))

            train_iter = mx.io.NDArrayIter(train, label_train_one_vs_all, batch_size, shuffle=True)
            val_iter = mx.io.NDArrayIter(test, label_test_one_vs_all, batch_size)

            # print(label_train[0:3])

            net = mx.sym.Variable('data')
            net = mx.sym.FullyConnected(net, name='fc1', num_hidden=nb_hidden_layer)
            net = mx.sym.Activation(net, name='relu1', act_type="relu")
            net = mx.sym.FullyConnected(net, name='fc2', num_hidden=nb_hidden_layer)
            net = mx.sym.SoftmaxOutput(net, name='softmax')
            mx.viz.plot_network(net)

            mod = mx.mod.Module(symbol=net, context=mx.cpu(), data_names=['data'], label_names=['softmax_label'])

#             #ml_metric = mx.metric.create(ml_acc)
#             #mx.metric.register(type(ml_metric), 'ml_metric')
            mod.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate': 0.1},
                    eval_metric='F1',
                    num_epoch=8)

            # predicted = mod.predict(val_iter)
            # val_iter.label[0][1].shape
            # val_iter.label
            score = mod.score(val_iter, ['acc', 'F1'])
            print("[Accuracy score, F1 score]: [%f, %f]" % (score[0][1], score[1][1]))


if __name__ == "__main__":
    train_example_csv = "../../out/train_example.csv"
    train_example_reduced_csv = "../../out/pca_train_example.csv"
    do_model(data_file_path=train_example_csv)
