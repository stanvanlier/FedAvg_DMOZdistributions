import torch
import numpy as np
import os
# from src.models import logreg_model
import torch.utils.data

def print_metrics(pred, target):
    import sklearn.metrics

    kappa = sklearn.metrics.cohen_kappa_score(pred, target)
    acc = sklearn.metrics.accuracy_score(target, pred)
    print("acc: {} |kappa: {} |".format(acc, kappa), end="")


def train_client(dataloader, model, criterion, optimizer, E, save_train=True):
    model.train()
    nk = len(dataloader.dataset)

    loss_all = np.zeros((len(dataloader), E,))
    y_y_pred = np.zeros((nk, E, 2,), dtype=int)

    for e in range(E):
        metrics_from_idx = 0
        for i_batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            # saving output for metrics
            if save_train:
                _, pred = output.topk(1, 1, True, True)
                pred.squeeze_()

                metrics_to_idx = metrics_from_idx + len(y)

                y_y_pred[metrics_from_idx:metrics_to_idx, e, 0] = y
                y_y_pred[metrics_from_idx:metrics_to_idx, e, 1] = pred

                metrics_from_idx = metrics_to_idx

                loss_all[i_batch, e] = loss.tolist()

    return y_y_pred, loss_all


def test_round(dataloader, model, criterion):
    model.eval()
    total_samples = len(dataloader.dataset)

    y_y_pred = np.zeros((total_samples, 2,), dtype=int)

    loss_all = np.zeros((len(dataloader),))
    metrics_from_idx = 0
    with torch.no_grad():
        for i_batch, (x, y) in enumerate(dataloader):
            output = model(x)

            loss = criterion(output, y)

            # saving output for matrics
            _, pred = output.topk(1, 1, True, True)
            pred.squeeze_()

            metrics_to_idx = metrics_from_idx + len(y)

            y_y_pred[metrics_from_idx:metrics_to_idx, 0] = y
            y_y_pred[metrics_from_idx:metrics_to_idx, 1] = pred

            metrics_from_idx = metrics_to_idx

            loss_all[i_batch] = loss.tolist()
    return y_y_pred, loss_all


def get_round_path(exp_path, t):
    return "{}/r{}".format(exp_path, t)


def prepare_path(path):
    os.makedirs(path, exist_ok=True)


def save_test_data(round_path, data):
    pred_mat, loss_arr = data
    np.savez_compressed("{}_test.npz".format(round_path), pred=pred_mat, loss=loss_arr)


def pad_zeros(x, seq_len):
    import numpy as np
    r = np.zeros((1, seq_len), dtype=np.long)
    r[0, :len(x)] = x
    return r


def make_dataloader(feature_df, B, shuffle=True, drop_last=False):
    SEQ_LEN = feature_df.len_x.max()

    X = torch.from_numpy(np.concatenate(feature_df.ids.apply(pad_zeros, args=(SEQ_LEN,)).values, axis=0)).long()
    y = torch.from_numpy(feature_df.y.values)

    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=shuffle, drop_last=drop_last)