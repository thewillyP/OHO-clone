import os, sys, math, time
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from metaopt.optimizer import SGD_Multi_LR, SGD_Quotient_LR
from itertools import product, cycle
import joblib
import wandb
from dataclasses import dataclass
import random
from mlp import *
from metaopt.util import *
from metaopt.util_ml import *

TRAIN = 0
VALID = 1
TEST = 2


@dataclass
class Config:
    project: str = "new_metaopt"
    test_freq: int = 10
    rng: int = 0
    dataset: str = "mnist"
    num_epoch: int = 100
    batch_size: int = 100
    batch_size_vl: int = 100
    model_type: str = "mlp"
    opt_type: str = "sgd"
    xdim: int = 784
    hdim: int = 128
    ydim: int = 10
    num_hlayers: int = 3
    lr: float = 1e-3
    mlr: float = 1e-4
    lambda_l1: float = 1e-4
    lambda_l2: float = 1e-4
    update_freq: int = 1
    reset_freq: int = -0
    beta1: float = 0.9
    beta2: float = 0.999
    valid_size: int = 10000
    checkpoint_freq: int = 10
    is_cuda: int = 0
    save: int = 0
    save_dir: str = "/scratch/ji641/imj/"


def save_object_as_wandb_artifact(obj, artifact_name: str, fdir: str, filename: str, artifact_type: str) -> None:
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"

    full_path = os.path.join(fdir, filename)
    joblib.dump(obj, full_path, compress=0)
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(full_path)
    wandb.log_artifact(artifact)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_mnist(args):
    dataset = datasets.MNIST(
        "data/mnist", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    train_set, valid_set = torch.utils.data.random_split(dataset, [60000 - args.valid_size, args.valid_size])

    data_loader_tr = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=0
    )
    data_loader_vl = DataLoader(
        valid_set, batch_size=args.batch_size_vl, shuffle=True, worker_init_fn=seed_worker, num_workers=0
    )
    data_loader_te = DataLoader(
        datasets.MNIST("data/mnist", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]
    return dataset


def main(args, trial=0, quotient=None, device="cuda"):
    dataset = load_mnist(args)

    hdims = [args.xdim] + [args.hdim] * args.num_hlayers + [args.ydim]
    num_layers = args.num_hlayers + 2
    if args.model_type == "amlp":
        model = AMLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=args.is_cuda)
        optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == "qmlp":
        model = QMLP(num_layers, hdims, args.lr, args.lambda_l2, quotient=quotient, is_cuda=args.is_cuda)
        optimizer = SGD_Quotient_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2, quotient=quotient)
    elif args.model_type == "mlp_drop":
        model = MLP_Drop(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=args.is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    else:
        model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=args.is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    print(
        "Model Type: %s Opt Type: %s Update Freq %d Reset Freq %d"
        % (args.model_type, args.opt_type, args.update_freq, args.reset_freq)
    )

    os.makedirs("%s/exp/mnist/" % args.save_dir, exist_ok=True)
    os.makedirs("%s/exp/mnist/mlr%f_lr%f_l2%f/" % (args.save_dir, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    fdir = "%s/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_fold%d/" % (
        args.save_dir,
        args.mlr,
        args.lr,
        args.lambda_l2,
        args.model_type,
        args.num_epoch,
        args.batch_size_vl,
        args.opt_type,
        args.update_freq,
        args.reset_freq,
        args.rng,
    )
    if quotient is not None:
        fdir = fdir.rstrip("/") + "_quotient%d/" % quotient

    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir + "/checkpoint/", exist_ok=True)
    args.fdir = fdir
    print(args.fdir)

    final_test_loss = train(args, dataset, model, optimizer, fdir)
    print("Final test loss %f" % final_test_loss)
    print(type(final_test_loss))
    return final_test_loss


def train(args, dataset, model, optimizer, fdir):
    counter = 0
    tr_loss_list, tr_acc_list, vl_loss_list = [], [], []
    optimizer = update_optimizer_hyperparams(args, model, optimizer)

    start_time0 = time.time()
    for epoch in range(args.num_epoch + 1):
        if epoch % 10 == 0:
            te_losses, te_accs = [], []
            for batch_idx, (data, target) in enumerate(dataset[TEST]):
                data, target = to_torch_variable(data, target, args.is_cuda, floatTensorF=1)
                _, loss, accuracy, _, _, _ = feval(data, target, model, optimizer, mode="eval", is_cuda=args.is_cuda)
                te_losses.append(loss)
                te_accs.append(accuracy)
            wandb.log(
                {
                    "test_epoch": epoch,
                    "test_loss": np.mean(te_losses),
                    "test_accuracy": np.mean(te_accs),
                }
            )
            print("Valid Epoch: %d, Loss %f Acc %f" % (epoch, np.mean(te_losses), np.mean(te_accs)))

        grad_list = []
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(dataset[TRAIN]):
            data, target = to_torch_variable(data, target, args.is_cuda)
            opt_type = args.opt_type
            model, loss, accuracy, output, noise, grad_vec = feval(
                data, target, model, optimizer, is_cuda=args.is_cuda, mode="meta-train", opt_type=opt_type
            )
            tr_loss_list.append(loss)
            tr_acc_list.append(accuracy)

            if args.reset_freq > 0 and counter % args.reset_freq == 0:
                model.reset_jacob()

            if counter % args.update_freq == 0 and args.mlr != 0.0:
                data_vl, target_vl = next(dataset[VALID])
                data_vl, target_vl = to_torch_variable(data_vl, target_vl, args.is_cuda)
                model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise)
                vl_loss_list.append(loss_vl.item())
                wandb.log(
                    {
                        "valid_epoch": counter,
                        "valid_loss": loss_vl.item(),
                    }
                )

            wandb.log(
                {
                    "train_epoch": counter,
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "learning_rate": model.eta,
                    "weight_decay": model.lambda_l2,
                    "dFdlr_norm": model.dFdlr_norm,
                    "dFdl2_norm": model.dFdl2_norm,
                    "grad_norm": model.grad_norm,
                    "grad_norm_vl": model.grad_norm_vl,
                    "grad_angle": model.grad_angle,
                    "param_norm": model.param_norm,
                }
            )

            grad_list.append(grad_vec)
            counter += 1

        corr_mean, corr_std = compute_correlation(grad_list, normF=1)
        wandb.log(
            {
                "corr_mean": corr_mean,
                "corr_std": corr_std,
            }
        )

        end_time = time.time()
        if epoch == 0:
            print("Single epoch timing %f" % ((end_time - start_time) / 60))

        if epoch % args.checkpoint_freq == 0:
            os.makedirs(fdir + "/checkpoint/", exist_ok=True)
            save_object_as_wandb_artifact(
                [model.state_dict()], f"model_{wandb.run.id}", f"{fdir}/checkpoint", f"epoch{epoch}", "model"
            )

        fprint = "Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f, Grad Corr %f %f"
        print(
            fprint
            % (
                epoch,
                np.mean(tr_loss_list[-100:]),
                np.mean(vl_loss_list[-100:]),
                np.mean(tr_acc_list[-100:]),
                str(model.eta),
                str(model.lambda_l2),
                model.dFdlr_norm,
                model.dFdl2_norm,
                model.grad_norm,
                model.grad_norm_vl,
                model.grad_angle,
                model.param_norm,
                corr_mean,
                corr_std,
            )
        )

    return te_losses[-1]


def feval(data, target, model, optimizer, mode="eval", is_cuda=0, opt_type="sgd", N=50000):
    if mode == "eval":
        model.eval()
        with torch.no_grad():
            output = model(data)
    else:
        model.train()
        optimizer.zero_grad()
        output = model(data)

    loss = F.nll_loss(output, target)
    pred = output.argmax(dim=1, keepdim=True).flatten()
    accuracy = pred.eq(target).float().mean()

    grad_vec = []
    noise = None
    if "train" in mode:
        loss.backward()

        for i, param in enumerate(model.parameters()):
            if opt_type == "sgld":
                noise = torch.randn(size=param.shape)
                if type(model.eta) == type(np.array([])):
                    eps = np.sqrt(model.eta[i] * 2 / N) * noise if model.eta[i] > 0 else 0 * noise
                else:
                    eps = np.sqrt(model.eta * 2 / N) * noise if model.eta > 0 else 0 * noise
                eps = to_torch_variable(eps, is_cuda=is_cuda)
                param.grad.data = param.grad.data + eps.data
            grad_vec.append(param.grad.data.cpu().numpy().flatten())

        if "SGD_Quotient_LR" in str(optimizer):
            optimizer.mlp_step()
        else:
            optimizer.step()
        grad_vec = np.hstack(grad_vec)
        grad_vec = grad_vec / norm_np(grad_vec)

    elif "grad" in mode:
        loss.backward()

    return model, loss.item(), accuracy.item(), output, noise, grad_vec


def meta_update(args: Config, data_vl, target_vl, data_tr, target_tr, model, optimizer, noise=None):
    param_shapes = model.param_shapes
    dFdlr = unflatten_array(model.dFdlr, model.param_cumsum, param_shapes)
    Hv_lr = compute_HessianVectorProd(model, dFdlr, data_tr, target_tr, is_cuda=args.is_cuda)

    # dFdl2 = unflatten_array(model.dFdl2, model.param_cumsum, param_shapes)
    # Hv_l2 = compute_HessianVectorProd(model, dFdl2, data_tr, target_tr, is_cuda=args.is_cuda)

    model, loss_valid, grad_valid = get_grad_valid(model, data_vl, target_vl, args.is_cuda)

    grad = flatten_array(get_grads(model.parameters(), args.is_cuda)).data
    param = flatten_array(model.parameters())
    model.grad_norm = norm(grad)
    model.param_norm = norm(param)
    grad_vl = flatten_array(grad_valid)
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_vl / model.grad_norm_vl).item()

    model.update_dFdlr(Hv_lr, param, grad, args.is_cuda, noise=noise)
    model.update_eta(args.mlr, val_grad=grad_valid)
    # param = flatten_array_w_0bias(model.parameters()).data
    # model.update_dFdlambda_l2(Hv_l2, param, grad, args.is_cuda)
    # model.update_lambda(args.mlr * 0.01, val_grad=grad_valid)

    optimizer = update_optimizer_hyperparams(args, model, optimizer)

    return model, loss_valid, optimizer


def get_grad_valid(model, data, target, is_cuda):
    val_model = deepcopy(model)
    val_model.train()

    output = val_model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    grads = get_grads(val_model.parameters(), is_cuda)
    model.grad_norm_vl = norm(flatten_array(grads))

    return model, loss, grads


def update_optimizer_hyperparams(args, model, optimizer):
    optimizer.param_groups[0]["lr"] = np.copy(model.eta)
    optimizer.param_groups[0]["weight_decay"] = model.lambda_l2
    return optimizer


if __name__ == "__main__":
    wandb_kwargs = {
        "mode": "offline",
        "group": "mnist_sweep",
        "config": Config().__dict__,
        "project": "new_metaopt",
    }

    with wandb.init(**wandb_kwargs) as run:
        args = Config(**run.config)
        seed = args.rng
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        main(args)
