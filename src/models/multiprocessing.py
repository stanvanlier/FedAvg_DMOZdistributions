from src.models import logreg_model, train_model
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import pandas as pd
import time

import pkg_resources

ROOT_DIR = pkg_resources.resource_filename("src", "..")

processed_path = os.path.join(ROOT_DIR, "data/processed")

SEQ_LEN = 165
EMBED_DIM = 50

train_df = pd.concat([pd.read_pickle(os.path.join(processed_path, "train_features_df.pkl")),
                      pd.read_pickle(os.path.join(processed_path, "train_distributions_df.pkl"))],
                     axis=1)
nodes_df = pd.read_pickle(os.path.join(processed_path, "nodes_df.pkl"))
K = len(nodes_df)
w_names = [n for n, p in logreg_model.LogReg(SEQ_LEN, EMBED_DIM).named_parameters() if p.requires_grad]
result_dir = os.path.join(ROOT_DIR, "results")

test_dataloader = train_model.make_dataloader(
    pd.read_pickle(os.path.join(processed_path, "test_features_df.pkl")),
    B=512, shuffle=False
)


def get_dataloader(dl, lock, k, d, B):
    lock.acquire()
    dataloader = dl[k]
    lock.release()
    
    if dataloader == None:
        distr = "d{}".format(d)
        node_train_df = train_df[train_df[distr] == k]
        
        dataloader = train_model.make_dataloader(node_train_df, B=B)
        lock.acquire()
        dl[k] = dataloader
        lock.release()
        
    return dataloader


def process(dl, lock, s_idx, s_lock,
            D_r, R, E, B, C):
    rank = dist.get_rank()
    m = int(max(round(C * K), 1))

    if rank == 0:
        os.nice(20)

    run_str = "avg-of-C"

    for d, r_start in D_r:
        
        np.random.seed(42+rank)
        torch.manual_seed(42+rank)
        
        model, criterion, optimizer, model_name = \
            logreg_model.make_LogRegAdam(lr=0.001, max_seqlen=SEQ_LEN, embed_dim=EMBED_DIM)

        exp_path = "{}/B{}_d{}/{}_{}/C{}_E{}".format(result_dir, B, d, run_str, model_name, C, E)

        
        # init run on process 0
        if rank == 0:
            train_model.prepare_path(exp_path)
            print(exp_path)

            # reset all global dataloaders
            for k in range(len(nodes_df)):
                dl[k] = None
            
            distr = "d{}".format(d)
            nk_arr = train_df[["y", distr]].groupby(distr).count().y
            
#             p_arr = [param for param in model.parameters() if param.requires_grad]
            
            for _ in range(r_start):
                np.random.choice(K, size=m, replace=False)
            
            if r_start > 0:
                round_path = train_model.get_round_path(exp_path, r_start)
                w = torch.load("{}_w_torch".format(round_path))
            else:
                w = {n:param for n, param in model.named_parameters() if param.requires_grad}
                
        else:
            w = {n:torch.zeros_like(param) for n, param in model.named_parameters() if param.requires_grad}

        for n in w_names:
            with torch.no_grad():
                dist.broadcast(w[n], src=0)

        for r in range(r_start+1, R+1):
            round_path = train_model.get_round_path(exp_path, r)

            S = torch.zeros(m, dtype=torch.long)
            ns = torch.tensor(0, dtype=torch.float)

            if rank == 0:
                train_model.prepare_path(round_path)
                S = np.random.choice(K, size=m, replace=False)
                ns.add_(float(nk_arr[S].sum()))
                S = torch.from_numpy(S).long()
                
                s_idx.set(0)
                rank0_train_counter = 0

            dist.broadcast(ns, src=0)
            dist.broadcast(S, src=0)
            
            if rank == 0:
                model.load_state_dict(w, strict=False)
                data = train_model.test_round(test_dataloader, model,criterion)
                y_y_pred, loss_all = data
                print("{} ({}) |loss: {} |".format(time.ctime()[11:-5], r-1, loss_all.mean()), end="")
                train_model.print_metrics(y_y_pred[:,1], y_y_pred[:, 0])
                train_model.save_test_data(train_model.get_round_path(exp_path, r-1), data)
                
            w_agg = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
            while True:
                s_lock.acquire()
                i = s_idx.get()
                s_idx.set(i+1)
                s_lock.release()
#                 print("\trank:{}, round:{}, s_idx:{}".format(rank,r,i),end=" ")
                if i >= m:
                    break
                if rank == 0:
                    rank0_train_counter += 1
                k = S[i]
#                 print("k:",k)
                dataloader = get_dataloader(dl, lock, k, d, B)
    
                model, criterion, optimizer, model_name = \
                    logreg_model.make_LogRegAdam(lr=0.001, max_seqlen=SEQ_LEN, embed_dim=EMBED_DIM)
                
                model.load_state_dict(w, strict=False)
            
                pred_mat, loss_mat = train_model.train_client(dataloader, model, criterion, optimizer, E)
                
                np.savez_compressed("{}/k{}_train.npz".format(round_path, k), pred=pred_mat, loss=loss_mat)
            
                f = float(len(dataloader.dataset))/ns
                p_arr = [param for param in model.parameters() if param.requires_grad]
                for i, p in enumerate(p_arr):
                    with torch.no_grad():
                        w_agg[i].add_(f, p)

            for n, p in zip(w_names, w_agg):
                dist.all_reduce(p, op=dist.reduce_op.SUM)
                p.requires_grad_(True)
                w[n] = p
            if rank == 0:
                print("{} trained ({}) on rank0: {}".format(time.ctime()[11:-5], r, rank0_train_counter))
            elif rank == 1 and r % 50 == 0:
                torch.save(w, "{}_w_torch".format(round_path))

        if rank == 0:
            model.load_state_dict(w, strict=False)
            round_path = train_model.get_round_path(exp_path, R)
            data = train_model.test_round(test_dataloader, model, criterion)
            y_y_pred, loss_all = data
            print("{} ({}) |loss: {} |".format(time.ctime()[11:-5], R, loss_all.mean()), end="")
            train_model.print_metrics(y_y_pred[:, 1], y_y_pred[:, 0])
            print("d{} done.".format(d))
            train_model.save_test_data(round_path, data)


def init_processes(rank, size, fn, run_kws):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('tcp', rank=rank, world_size=size)
    fn(**run_kws)


def run(C=0.1, d_startRound_arr=[(0, 0), (25, 0)], rounds=250, client_epochs=5, batch_size=64,
        num_proc=2):
    m = int(max(round(C * K), 1))

    # SAVE_TRAIN = True
    # EMBED_DIMS = 50

    M = mp.Manager()
    dl = M.list()
    lock = M.Lock()
    s_lock = M.Lock()
    s_idx = M.Value("i", 0, lock=True)

    run_kws = dict(dl=dl,
                   lock=lock,
                   s_idx=s_idx,
                   s_lock=s_lock,
                   D_r=d_startRound_arr,
                   R=rounds,
                   E=client_epochs,
                   B=batch_size,
                   C=C)
                   # m=m,)

    print("{}|processes: {}, C: {}, m: {}, dir: {}".format(time.ctime(), num_proc, C, m, result_dir))
    print(run_kws)

    for k in range(len(nodes_df)):
        dl.append(None)

    processes = []
    for rank in range(num_proc):
        p = mp.Process(target=init_processes, args=(rank, num_proc, process, run_kws))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()