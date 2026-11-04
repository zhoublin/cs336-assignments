import os
import argparse
import numpy as np
import torch
from datetime import datetime

from .bpe import BPETrainer, Tokenizer
from .loss import cross_entropy, AdamW, lr_cosine_schedule, gradient_clipping
from .transformer import TranformerLM
from .dataset import data_loading, save_checkpoint, load_checkpoint
from .plogger import ProjectLogger
from .utils import plot_loss


logger = ProjectLogger("TransformerLM training").logger

def get_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def train_tokenizer(args):
    logger.info(f"train BPE tokenizer began at {get_time_now()}")
    trainer = BPETrainer(
        input_path=args.dataset_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens
    )
    trainer.train()
    logger.info(f"train BPE tokenizer finished at {get_time_now()}")

    return Tokenizer(
        vocab=trainer.get_vocab(),
        merges=trainer.get_merges(),
        special_tokens=args.special_tokens
    )

def prepare_dataset(
        args,
        tokenizer,
        save_path="./dataset.bin",
        dtype=np.uint32
):
    with open(args.dataset_path, 'rb') as f:
        token_ids = [item for item in tokenizer.encode_iterable(f)]

    np.save(save_path, np.array(token_ids, dtype=dtype))
    return save_path

def train_model(args, data):
    d_ff = args.d_ff if (args.d_ff != -1) \
                     else (round((8 / 3 * args.d_model) / 64) * 64)
    
    model = TranformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
    )
    model.train()

    optimizer = AdamW(
        params=model.parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    min_loss = 100.0
    train_losses = list()
    for iter in range(args.steps):
        (x_seq, y_seq) = data_loading(
            dataset=data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )
        out = model(x_seq)
        loss = cross_entropy(out, y_seq)
        loss.backward()
        # 梯度裁剪
        gradient_clipping(model.parameters, args.max_l2_norm)
        optimizer.step()

        train_losses.append(loss.item())
        if loss.item() < min_loss:
            save_checkpoint(
                model,
                optimizer,
                iter,
                os.path.join(args.checkpoint_path, "best.ckpt")
            )

        # update lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_cosine_schedule(
                it=iter,
                max_learning_rate=args.lr_max,
                min_learning_rate=args.lr_min,
                warmup_iters=args.warmup_iter,
                cosine_cycle_iters=args.cycle_iter
            )

    plot_loss(
        train_losses,
        os.path.join(args.output_path, "train_loss.png"),
        "Training loss"
    )
    return model, optimizer

def valid_model(args, model, data):
    model.eval()
    eval_losses = []
    for iter in range(args.eval_steps):
        (x_seq, y_seq) = data_loading(
            dataset=data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )
        out = model(x_seq)
        loss = cross_entropy(out, y_seq)
        eval_losses.append(loss.item())

    plot_loss(
        eval_losses,
        os.path.join(args.output_path, "eval_loss.png"),
        "Evaluation loss"
    )

def main(args):
    # 0. 训练tokenizer
    tokenizer = train_tokenizer(args)

    # 1. 准备数据
    save_path = prepare_dataset(args, tokenizer)
    data = np.load(save_path, mmap_mode="r")

    # 2. 训练模型
    model, optimizer = train_model(args, data)

    iteration = load_checkpoint(
        src=os.path.join(args.checkpoint_path, "best.ckpt"),
        model=model,
        optimizer=optimizer
    )

    # 3. 测试模型
    valid_model(args, model, data)


def parse_arguments():
    parser = argparse.ArgumentParser("Transformer LM training script (GPT-2 XL-sized default).")
    parser.add_argument("-p", "--dataset_path", required=True, type=str, help="sepcify the dataset path.")
    parser.add_argument("-d", "--device", default="cuda", type=str, help="CPU ('cpu') or GPU ('cuda').")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size for training.")
    parser.add_argument("-t", "--steps", default=64, type=int, help="the num of training steps.")

    # tokenzier / model
    parser.add_argument("-s", "--special_tokens", type=str, nargs="+", help="special tokens in vocabulary.")
    parser.add_argument("-v", "--vocab_size", default=50257, type=int, help="vocab size for BPE tokenizer training.")
    parser.add_argument("-c", "--context_length", default=1024, type=int, help="context length.")
    parser.add_argument("-n", "--num_layers", default=48, type=int, help="specify the num of transformer blocks.")
    parser.add_argument("-m", "--d_model", default=1600, type=int, help="embedding dimension of transformer blocks.")
    parser.add_argument("-h", "--num_heads", default=25, type=int, help="the num of attention heads.")
    parser.add_argument("-f", "--d_ff", default=-1, type=int, help="hidden representation dimension of FFN (-1 means 8/3 * d_model).")

    # optimizer
    parser.add_argument("-lr", "--learning_rate", deafult=1e-3, type=float, help="learning rate for training.")
    parser.add_argument("-wd", "--weight_decay", default=.0, type=float, help="weight_decay in AdamW optimizer.")
    parser.add_argument("-b1", "--beta1", default=0.9, type=float, help="first beta param in Adamw optimizer.")
    parser.add_argument("-b2", "--beta2", default=0.999, type=float, help="second beta param in Adamw optimizer.")
    parser.add_argument("-eps", "--epsilon", default=1e-8, type=float, help="epsilon in Adamw optimizer.")

    # schedule
    parser.add_argument("--max_lr", default=1e-3, type=float, help="max learning rate for cosine schedule.")
    parser.add_argument("--min_lr", default=1e-7, type=float, help="min learning rate for cosine schedule.")
    parser.add_argument("--warmup_iter", default=10, type=int, help="warmup steps for cosine schedule.")
    parser.add_argument("--cycle_iter", default=20, type=int, help="cosine cycle began step for cosine schedule.")
    
    # gradient
    parser.add_argument("--max_l2_norm", default=1e-1, type=float, help="max l2 norm for gradient clipping.")

    # other
    parser.add_argument("--output_path", default="output", type=str, help="output path for saving pics and logs.")
    parser.add_argument("--checkpoint_path", default="models", type=str, help="output path for saving checkpoints.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)