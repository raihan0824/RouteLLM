import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
import safetensors.torch
import json 
from routellm.routers.matrix_factorization.model import MODEL_IDS

# Initialize the OpenAI API client with your API key
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.models_a = torch.tensor(
            [MODEL_IDS[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [MODEL_IDS[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompts = [sample["prompt"] for sample in data]
        self.winners = [sample["winner"] for sample in data]

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        assert self.winners[index] in ["model_a", "model_b"], self.winners[index]
        if self.winners[index] == "model_a":
            return self.models_a[index], self.models_b[index], self.prompts[index]
        else:
            return self.models_b[index], self.models_a[index], self.prompts[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)

class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models,
        num_prompts,
        text_dim=3584,
        num_classes=1,
        use_proj=True,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.prompt_to_embedding = {}  # Store embeddings for each prompt

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = nn.Linear(
            dim, num_classes, bias=False
        )  # bias should be False!

    def get_device(self):
        return self.P.weight.device

    def get_prompt_embedding(self, prompt):
        if prompt not in self.prompt_to_embedding:
            response = client.embeddings.create(
                input=[prompt[:3000]],
                model="baai/bge-multilingual-gemma2"
            )
            self.prompt_to_embedding[prompt] = torch.tensor(response.data[0].embedding, device=self.get_device())
        return self.prompt_to_embedding[prompt]

    def forward(self, model_win, model_loss, prompt, test=False, alpha=0.05):
        model_win = model_win.to(self.get_device())
        model_loss = model_loss.to(self.get_device())

        model_win_embed = self.P(model_win)
        model_win_embed = F.normalize(model_win_embed, p=2, dim=1)
        model_loss_embed = self.P(model_loss)
        model_loss_embed = F.normalize(model_loss_embed, p=2, dim=1)
        prompt_embed = self.get_prompt_embedding(prompt[0])
        if not test:
            # adding noise to stablize the training
            prompt_embed += torch.randn_like(prompt_embed) * alpha
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(
            (model_win_embed - model_loss_embed) * prompt_embed
        ).squeeze()

    @torch.no_grad()
    def predict(self, model_win, model_loss, prompt):
        logits = self.forward(model_win, model_loss, prompt, test=True)
        return logits > 0

def evaluator(net, test_iter, device):
    net.eval()
    ls_fn = nn.BCEWithLogitsLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models_a, models_b, prompts in test_iter:
            # Assuming devices refer to potential GPU usage
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts

            logits = net(models_a, models_b, prompts)
            labels = torch.ones_like(logits)
            loss = ls_fn(logits, labels)  # Calculate the loss
            pred_labels = net.predict(models_a, models_b, prompts)

            # update eval stats
            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())
            num_samples += labels.shape[0]

    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples

def train_loops(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=evaluator,
    **kwargs,
):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss(reduction="mean")

    def train_epoch():  # Inner function for one epoch of training
        net.train()  # Set the model to training mode
        train_loss_sum, n = 0.0, 0
        for models_a, models_b, prompts in train_iter:
            # Assuming devices refer to potential GPU usage
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts

            output = net(models_a, models_b, prompts, alpha=alpha)
            ls = loss(output, torch.ones_like(output))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            train_loss_sum += ls.item() * len(models_a)
            n += len(models_a)
        return train_loss_sum / n

    train_losses = []
    test_losses = []
    test_acces = []
    best_test_acc = -1
    progress_bar = tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        train_ls = train_epoch()
        train_losses.append(train_ls)
        info = {"train_loss": train_ls, "epoch": epoch}

        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, device)
            test_losses.append(test_ls)
            test_acces.append(test_acc)
            info.update(
                {
                    "test_loss": test_ls,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "best_test_loss": min(test_losses),
                }
            )
        else:
            test_ls = None  # No evaluation

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        progress_bar.set_postfix(**info)
        progress_bar.update(1)

    progress_bar.close()

if __name__ == "__main__":
    dim = 128
    batch_size = 64
    num_epochs = 10
    alpha = 0.1
    use_proj = True
    lr = 3e-4
    weight_decay = 1e-5

    # load and filter data
    data_raw = load_dataset(path="lmsys/lmsys-arena-human-preference-55k", split="train").to_pandas()
    
#     dataset1 = load_dataset(path="lmsys/lmsys-arena-human-preference-55k", split="train")
#     dataset2 = load_dataset(path="routellm/gpt4_judge_battles", split="train")

#     # Concatenate the datasets
#     data_raw = concatenate_datasets([dataset1, dataset2]).to_pandas()

    def preprocess_data(battles_df):
        MIN_LEN = 16

        def get_first_turn(prompt_str):
            return json.loads(prompt_str)[0].strip()

        def get_winner(row):
            if row["winner_model_a"] == 1:
                return "model_a"
            elif row["winner_model_b"] == 1:
                return "model_b"
            else:
                return "tie"

        battles_df["prompt"] = battles_df["prompt"].apply(get_first_turn)
        battles_df["winner"] = battles_df.apply(get_winner, axis=1)
        battles_df = battles_df.loc[battles_df["prompt"].apply(len) >= MIN_LEN]
        battles_df = battles_df[["prompt","model_a", "model_b", "winner"]]

        return battles_df.to_dict(orient='records')
    
    data = preprocess_data(data_raw)

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]
    # shuffle and prepare train test split
    data_shuffled = filtered_data.copy()
    random.shuffle(data_shuffled)
    train_data = data_shuffled[: int(len(data_shuffled) * 0.95)]
    test_data = data_shuffled[int(len(data_shuffled) * 0.95) :]

    train_data_loader = PairwiseDataset(train_data).get_dataloaders(
        batch_size=batch_size, shuffle=True
    )
    test_data_loader = PairwiseDataset(test_data).get_dataloaders(1024, shuffle=False)

    model = MFModel_Train(
        dim=dim,
        num_models=len(MODEL_IDS),
        num_prompts=len(filtered_data),
        use_proj=use_proj,
    ).to("cuda")

    train_loops(
        model,
        train_data_loader,
        test_data_loader,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        num_epochs=num_epochs,
        device="cuda",
    )
    safetensors.torch.save_file(model.state_dict(), "model.safetensors")