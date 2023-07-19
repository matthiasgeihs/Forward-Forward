import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src import utils
from src.ff_model import FF_model


def train(opt, model: FF_model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        print(f"Epoch: {epoch}")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)
            
        # Generate.
        context = """First Citizen:
Before we proceed any further, hear me speak. No """
        tokens = torch.Tensor([train_loader.dataset.stoi[x] for x in context]).unsqueeze(0).to(opt.device)
        generated = model.generate(tokens)
        print("".join([train_loader.dataset.itos[x.item()] for x in generated[0]]))

    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    my_main()
