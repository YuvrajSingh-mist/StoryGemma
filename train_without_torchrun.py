
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm 
from dataclasses import dataclass
from torch.nn import RMSNorm
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
# from torchtune.modules import RMSNorm
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetDecoderModelOutput
import wandb
from tqdm import tqdm
from functools import partial

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# import wandb
# wandb.login()


# from torch.utils.tensorboard import SummaryWriter


from datasets import load_dataset

tinystories = True
fw = False
fw_train = None
fw_test = None
if(tinystories):
    fw_train = load_dataset("roneneldan/TinyStories", split="train")
    fw_test = load_dataset("roneneldan/TinyStories", split="validation")
    print(fw_train)
    print(fw_test)
if(fw):   
    fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
    fw_train = fw_train.train_test_split(test_size=0.01)
    print(fw_train)
    # print(fw_test)


def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()



# Load model directly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", token='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')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Tokenizer!")


@dataclass
class ModelArgs:
    #Hyperparameters

    block_size = 512
    batch_size = 256
    embeddings_dims = 512
    attn_dropout = 0.1
    no_of_heads = 8 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 100
    max_lr = 2.5e-4
    no_of_decoder_layers = 8 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    device = 'cuda:6'
    no_kv_heads = 2
    scaling_factor = 0.5
    vocab_size = len(tokenizer.get_vocab()) + 768
    local_block_size = 128
    base_freq=10000
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    clip = 1.0
    
    
    
from pathlib import Path
data_path = Path('data')
data_path.mkdir(exist_ok=True)




def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "SCHEDULER_STATE": scheduler.state_dict(),  # NEW: Save scheduler state
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, "snapshot_3.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    # scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])  # Load scheduler state
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step



def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=ModelArgs.block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)
    # alpaca_prompt = '''
    

    def collate_fn(batch):
        # Extract text data
        texts = [item ["text"] for item in batch]

    
        input_encodings = tokenizer(texts, padding='max_length', max_length=ModelArgs.block_size, truncation=True, return_tensors="pt")
    
        input_encodings["labels"] = input_encodings["input_ids"].clone()  # Use `input_ids` as labels
        
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  # Shift right
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end 
    
        return input_encodings

    
    dataloader = None
    if(tinystories):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train,
            # generator=generator,
            batch_size=batch_size,
            
            # sampler=DistributedSampler(fw_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            #  pin_memory=True,  # Add this
            # persistent_workers=True
        )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_test,

            batch_size=batch_size,
            # sampler=DistributedSampler(fw_test, shuffle=False),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False, 
            # pin_memory=True,  # Add this
            # persistent_workers=True
        )
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            
            
            # sampler=DistributedSampler(fw_train['train'], shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            # num_workers=os.cpu_count(),
            num_workers = min(4, os.cpu_count()//2),  # Don't overallocate
            prefetch_factor = 2,  # Balance memory/performance       
            pin_memory=True,  # Add this
            persistent_workers=True
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
                # generator=generator,
            # sampler=DistributedSampler(fw_train["test"], shuffle=False),
            collate_fn=collate_fn,
            # num_workers=os.cpu_count(),
            num_workers = min(4, os.cpu_count()//2), # Don't overallocate
            prefetch_factor = 2,  # Balance memory/performance
            drop_last=True,
            shuffle=False,
            pin_memory=True,  # Add this
            persistent_workers=True
        )
    return data_loader


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])
    x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    return x, y

from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size  # Ensure valid indexing

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



def create_sequences(data, block_size):
    sequences = []

    for seq in data:
        len(seq)
        if len(seq) < block_size:
            # while(len(sequence) < block_size):
                # sequence = data[i:i + block_size + 1]
        
                # Pad the sequence if it's shorter than block_size
            padding_length = block_size - len(seq)
            seq = torch.cat([seq, torch.full((padding_length,), tokenizer.encode('[PAD]').ids[0], dtype=torch.long)])

        else:
            if len(seq) > block_size:
                seq = seq[:block_size]
            # while(len(sequence) < block_size):
                # sequence = data[i:i + block_size + 1]
        
                # Pad the sequence if it's shorter than block_size
            # padding_length = block_size - len(seq)
            # seq = torch.cat([seq, torch.full((padding_length,), tokenizer.encode('[PAD]').ids[0], dtype=torch.long)])
        sequences.append(seq)
    out = torch.tensor(sequences, dtype=torch.long)
    return out

# train_data = create_sequences(train_data_flat['input_ids'], ModelArgs.block_size)
# val_data = create_sequences(val_data['input_ids'], ModelArgs.block_size)


# Define collate_fn
def collate_fn(split , batch):
    block_size = ModelArgs.block_size
    batch_size = len(batch)
    if(split == 'train'):
        data = train_data_tensor
    elif(split == 'test'):
        data = val_data_tensor
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])

    # print("Shape of x: ", len(x))
    # print("Length of y: ", len(y))
    # x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    # x = torch.zeros((batch_size, block_size), dtype=torch.long)
    # y = torch.zeros((batch_size, block_size), dtype=torch.long)
    # for i, sequence in enumerate(batch):
    #     print("Seq: ", sequence)
    #     print("Shape x: ", sequence[:-1].shape)
    #     print("Shape of y: ", len(sequence[1:]))
    #     x[i] = sequence[:-1]  # Input is all tokens except the last one
    #     y[i] = sequence[1:]   # Target is all tokens except the first one
    return x, y
    

class Normalization(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):  
        super().__init__()
        self.rmsnorm_layer = RMSNorm(normalized_shape=embeddings_dims)
        
        
    def forward(self, x):
        
        x = self.rmsnorm_layer(x)
        return x
        

# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
        device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size,
        scaling_factor: float = 0.5,
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.scaling_factor = scaling_factor
        self.theta = 0
        self.device=device

    def apply_rope(self, seq, base_freq):
        batch_size, seq_len, embeds_dims = seq.shape
        token_indices = torch.arange(0 , seq_len, dtype=torch.float32,  device = self.device).unsqueeze(1)
        positions = torch.arange(0 , self.embeddings_dims, 2, dtype=torch.float32,  device = self.device).unsqueeze(0)
        theta = base_freq ** (-2 * (positions * self.scaling_factor) / self.embeddings_dims) #Position Interpolation
        angles = token_indices * theta
        angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
        x_reshaped = seq.view(batch_size, seq_len, self.embeddings_dims // 2, 2)
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)


        out = torch.stack([x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles], dim=1)
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x, base_freq):

        res = self.apply_rope(x,base_freq=base_freq)
        return res 
    

class MQA(nn.Module):
    def __init__(
        self,
        device,
        no_of_q_heads: int,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        

    ):
        super().__init__()


        # self.no_of_q_heads = no_of_heads // no_of_kv_heads
        # self.no_of_q_heads = no_of_q_heads
        self.no_of_kv_heads = 2 # I want to have a kv for each pair of query heads 
        self.head_size = embeddings_dims // no_of_q_heads
        # self.kv_head_size = (embeddings_dims // self.no_of_kv_heads) * 2
        self.rotary= RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        # self.rotary_k = RotaryEmbeddings(embeddings_dims=self.kv_head_size,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=self.head_size * self.no_of_kv_heads, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = self.device) for _ in range(self.no_of_kv_heads)])

    def scaled_dot_product(self, q, k, v, block_size, base_freq):

            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            normalized_q = q * (torch.norm(q, p=2)** -1)
            q = self.rotary(normalized_q, base_freq)
            masked_table = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            # rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # print("Query: ", q.shape)
            # print("Keys: ", k.shape)
            # print(q.permute(2,0,1).shape)
            # print(k.permute(2,0,1).transpose(-2, -1).shape)
            # weights = q.permute(2,0,1) @ k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            # weights = q @ k.permute(2,1,0)
            # print(weights.shape)
            # print(masked.shape)
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out

    def forward(self,x, base_freq=10000):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape

        # query = self.query(x)
        # matrix = self.rotary_matrix(block_size)


        key = self.key(x)
        key_normalized = key * (torch.norm(key, p=2)** -1)
        values = self.value(x)
        # print("Keys: ", key.shape)
        # print("Values: ", values.shape)
        # rotary_value = self.rotary(values)
        rotary_key = self.rotary(key_normalized, base_freq)
        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), rotary_key, values, block_size, base_freq) for query in self.multi_query], dim=-1)
        # print("Multi query: ", multi_query_concat.shape)

        linear_layer= self.linear_layer(multi_query_concat)
        # out = self.dropout(linear_layer)
        return linear_layer
    
class GQA(nn.Module):
    def __init__(
        self,
        device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        # no_of_q_heads: int = ModelArgs.no_of_heads,
        mqa_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        # self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = ModelArgs.no_of_heads // mqa_heads
        # self.head_dim = embeddings_dims // self.no_kv_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_q_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(no_of_q_heads=self.no_of_q_heads, embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_q_heads)])
        # self.mqa = MQA(no_of_q_heads=self.no_of_q_heads, device=self.device, embeddings_dims=embeddings_dims, block_size=block_size)
    def forward(self,x, base_freq):

        batch, block_size, embeddings_dims = x.shape

        # res = self.mqa(x)
        grouped_query_concat = torch.cat([group(x, base_freq) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat) #Basically MQA is made into GQA with no_of_q_heads and this class right here is just to consolidate everything into one
        out = self.dropout(linear_layer)
        return out

    

class Swish(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish



class SWiGLU(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims*2,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims*2,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=embeddings_dims*2, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out



class FFN(nn.Module):
    def __init__(self,
                device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,
                dropout = ModelArgs.dropout

                ):
        super().__init__()

        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        # self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        # x = self.linear_layer(x)
        # x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                ) :
        super().__init__()

        # self.base_freq = ModelArgs.base_freq
        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, mqa_heads=2,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x, base_freq):

        x = x + self.gqa(self.norm1(x), base_freq)
        x = x + self.feedforward_network(self.norm2(x))
        return x

class Gemma3(nn.Module):
    def __init__(self,
                    device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,
                dropout = ModelArgs.dropout

                ) :
        super().__init__()
        self.base_freq = ModelArgs.base_freq
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.ModuleList(DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers))
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = Normalization(embeddings_dims)
        
        
        #weight tying
        # self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
                    
                    
    def forward(self, x):
        global_base_freq = 100000 
        local_base_freq = 10000
        index = 0
        no_of_layers = 0
        x = self.embeddings(x)
        x = self.dropout(x)
        # temp = x.clone()
        # x = self.decoder(x)
        for layer in self.decoder:
            if no_of_layers % 5 == 0:
                x = layer(x, global_base_freq)
                # print("x shape: ", x.shape)
            else:
                
                local_block = x[:, : index + ModelArgs.local_block_size, :]
                x = layer(local_block, local_base_freq)
                index += ModelArgs.local_block_size
                # print("x shape local: ", x.shape)
            no_of_layers += 1
        # print(x.shape)
        x = self.norm(x)
        x = self.linear_layer(x)
        
        return x


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def greedy_decode(
    model, 
    tokenizer, 
    prompt, 
    max_length=50, 
    repetition_penalty=1.2, 
    context_window=10, 
    temperature=1.0, 
    eos_token_id=None
):

    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    generated_tokens = []
    eos_token_id = eos_token_id or tokenizer.eos_token_id  # Use EOS token if provided

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs[:, -1, :]  # Get logits for the last token

        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for token in set(generated_tokens[-context_window:]):  # Penalize recent tokens
                logits[0, token] /= repetition_penalty

        # Greedy selection
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated_tokens.append(next_token.item())

        # Stop if EOS token is generated
        if next_token.item() == eos_token_id:
            break

        # Append the new token to the input
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the generated tokens
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)



def save_to_file(step, text):
    
    with open('generations.txt', 'a') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")
        
    
#Train the  model


# writer = SummaryWriter(log_dir="runs/experiment")

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# Warmup phase for 2000 steps
def warmup_fn(step):
    if step < 2000:
        return step / 2000  # LR gradually increases
    return 1.0





#Train the  model



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention

torch.set_float32_matmul_precision('high')

# scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))

save_checkpoint_iter = 1000
total_iters = 10000
eval_iters = 500 #should be at 1000
eval_check = 100
warmup_iters = 1000
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 10000
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * torch.cuda.device_count()))

# learning rate decay scheduler (cosine with warmup) from https://github.com/karpathy/nanoGPT/blob/master/train.py



# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids_len = len(input_ids[0])
    
    generated_tokens = []
    # ModelArgs.inference=True
    for _ in range(max_length - input_ids_len):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            
            # Apply temperature scaling
            probs = probs / temperature
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
        
            
            # generated_tokens.append(next_token.item())
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            # generated_tokens.append(xcol)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence

            # if(next_token.item() == tokenizer.eos_token_id):
            #     break
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)





def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (ModelArgs.max_lr - min_lr)



model = Gemma3(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
model = model.to(ModelArgs.device)

# Printing a summary of the architecture
# !pip install torchinfo
from torchinfo import summary
# idx, targets = get_batch('test')
idx = torch.randint(
        low=0,
        high=ModelArgs.vocab_size,
        size=(ModelArgs.batch_size, ModelArgs.block_size),
        dtype=torch.long
    )
# sample_idx = random.randint(range(len(train_dataset)))
# idx, targets = train_dataset[0]
idx = idx.to(ModelArgs.device)
# targets = targets.to(ModelArgs.device)
summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
    
    
        row_settings=["var_names"])




class CustomLRScheduler:
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.it = 0
        self._last_lr = [max_lr]  # Initialize with max_lr (matching PyTorch convention)
        
    def step(self):
        
        self._last_lr = [self._get_lr()]  # Store as list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._last_lr[0]
        self.it += 1

    def get_last_lr(self):
        return self._last_lr  # Returns list to match PyTorch convention
    
    def _get_lr(self):

        # cycle = math.floor(1 + self.it / (2 * self.warmup_iters))
        # x = abs(self.it / self.warmup_iters - 2 * cycle + 1)
        # return self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x))
        
        # 1) linear warmup for warmup_iters steps
        if self.it < self.warmup_iters:
            return self.max_lr * (self.it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if self.it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'lr_decay_iters': self.lr_decay_iters,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'it': self.it
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.lr_decay_iters = state_dict['lr_decay_iters']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.it = state_dict['it']


scaler  = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))

def train():
    # setup()
    # device = int(os.environ["LOCAL_RANK"])
    # device = 0
    # torch.cuda.set_device(int(device))
    # torch.set_default_device('cuda')
    # train_dataloader = prepare_dataset(ModelArgs.batch_size)
    # rank = torch.distributed.get_rank()
    # print(f"Start running DDP on rank {device}.")
    # # create model and move it to GPU with id rank
    # device_id = rank % torch.cuda.device_count()
    # CFG = ModelArgs()

    # if(device == 0):
    print("dtype: ", ModelArgs.dtype)
    device=ModelArgs.device
    
#         # Initialise run
    wandb.init(
            # entity = 'rajceo2031',
                        project = 'Mixtral-DDP-Pretrain-10-billion-tokens',
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    print("wandb initialized")
    model = Gemma3(device=ModelArgs.device, embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)

    # print(f"Model on device {device} is ready")
    # print(f"Model on device {device} is ready")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=ModelArgs.max_lr, 
        betas=(ModelArgs.beta_1, ModelArgs.beta_2),
        weight_decay=ModelArgs.weight_decay_optim,
        # eps=ModelArgs.eps,
        fused=True
    )
    # if(ModelArgs.use_compile):
    #     model = torch.compile(model,  mode='max-autotune')

    model = model.to(device)
    
    # model = DDP(model, device_ids=[device])
    

    
    
    model.eval()
    world_size = torch.cuda.device_count()
    @torch.inference_mode()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', ModelArgs.batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            # if(split == 'train'):
            #         loader = train_loader
            # if(split == 'val'):
            #         loader = val_loader
            for step in range(eval_check):  
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    batch = next(val_loader_iterator)
                
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0 
                # batch = next(val_loader_iterator)
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids']
                targets = batch['labels']
                idx = idx.to(device)
                targets = targets.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # if(ModelArgs.use_liger):
                        # loss = model(idx, actual_labels = targets)
                    # else:
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    # print(logits.shape)   
                    # print(targets)
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    # print("OK")
                    targets = targets.view(batch_size * block_size)
                    # print("OK2")
                    loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                    # batch_size, block_size, embeddings_dims = logits.shape
                    # logits = logits.view(batch_size * block_size, embeddings_dims)
                    # targets = targets.view(batch_size * block_size)

                    # loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                total_loss += loss.item()
                total_batches += 1

            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

            # Compute mean loss across all evaluation epochs
            out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            epoch_loss = None
            epoch_losses = []

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    count = 0
   
    train_dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
    val_loader= prepare_dataset('val', device, ModelArgs.batch_size)
    # for step in tqdm(range(total_iters)):
    # for epoch in range(ModelArgs.epochs):
        # torch.cuda.synchronize() 
    
    # train_dataloader.sampler.set_epoch(epoch)
    
    # val_loader.sampler.set_epoch(epoch)
    print("Loaders ready both")
    epochs = ModelArgs.epochs
    scheduler = CustomLRScheduler(
    optimizer,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    max_lr=ModelArgs.max_lr
)
    # train_step_iterator = range(len(train_dataloader))
    # if device == 0:  # Only create progress bar on rank 0
    #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

        # Print progress on rank 0
    train_loader_length = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
    # if(device == 0):
        # train_loader_length = len(train_dataloader)
        # print("Total batches: ", train_loader_length)
    # print("Length of : ", len(train_dataloader))
    # print("Length of val: ", len(val_loader))
    # for  step, batch in enumerate(train_dataloader):
    for step in tqdm(range(total_iters)):
        # print("Dataloader things: ", batch)
        # print("Total batches: ", len(train_dataloader))
        
        
        # if(device == 'cuda:0'):
    # if(step % 100 == 0):
#     if(step == train_loader_length):
#       break
        print("Step : ", step, "/", total_iters)
        # print('Total batches: ', len(train_dataloader))
        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
        print("Total tokens processed: ", token_count)
        
        # all_gpus_avg_train_loss = None
        # all_gpus_avg_val_loss = None
        # every once in a while evaluate the loss on train and val sets
        if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
            losses = estimate_loss( val_loader, val_data_iterator, 'cuda')
            # avg_train_loss = losses['train']
            avg_val_loss = losses['val']
            # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # if device == 0:  # Only print on main process
            print(f"[GPU {device}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
            # print(f"[GPU {device}] | Epoch {epoch}/{ModelArgs.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f}")
                # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # Log training loss more frequently
                # Aggregate average loss across all GPUs
            # avg_train_loss = torch.Tensor([losses['train']]).to(device)
            avg_val_loss = torch.Tensor([losses['val']]).to(device)
            # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            # torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            # if device == 'cuda:0':
                # all_gpus_avg_train_loss = avg_train_loss / world_size
                # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
            all_gpus_avg_val_loss = avg_val_loss / world_size
            print(f"Val Loss: {all_gpus_avg_val_loss.item():.4f}")
            
      
            
            perplexity = torch.exp(torch.tensor(all_gpus_avg_val_loss.item()))  # Calculate perplexity

            # if device == 0:
            wandb.log({
                    "All GPU Val_Loss": all_gpus_avg_val_loss.item(),
                    "Val Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                })
            print(f"Step: {step} | All GPU Val Loss: {all_gpus_avg_val_loss.item():.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")
            
            


        if step % save_checkpoint_iter == 0 and step != 0:
            print(f"Saving the model checkpoint for step: {step}")
            _save_snapshot(model, optimizer, None, None, step)
        
        accumulated_loss = 0.0
        
        
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                batch = next(train_data_iterator)

            idx = batch['input_ids'].to(device)

            targets = batch['labels'].to(device)
            token_count += (len(idx) * ModelArgs.batch_size)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # if(ModelArgs.use_liger):
                #     loss = model(idx, actual_labels = targets)
                # else:
                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                # print(logits.shape)   
                # print(targets)
                logits = logits.view(batch_size*block_size, embeddings_dims)
                # print("OK")
                targets = targets.view(batch_size * block_size)
                # print("OK2")
                loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                
            loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
            accumulated_loss += loss.detach()
            # loss.backward() 
            scaler.scale(loss).backward()
        # model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
        # scaler.scale(loss).backward()
        # Check for unused parameters
    # del logits, targets, loss
    

        unused_params = find_unused_parameters(model)
        if unused_params:
            print(f"Unused parameters: {unused_params}")
    # break

        # if(device == 0):
            # if(micro_step % 10 == 0):
        #     if(step == train_loader_length):
        #       break
                
            # print("Micro Batch : ", micro_step)
        print("Step : ", step, "/", total_iters)
            # print('Total batches: ', len(train_dataloader))
            # print("Total gradient accumulation steps: ", gradient_accumulation_steps)
        print("Total tokens processed: ", token_count)
        # count += 1
       
        # lr = cyclical_lr(step)
        # for params in optimizer.param_groups:
        #     params['lr'] = lr
            
        
        
        # Compute gradient norms before clipping
        if(ModelArgs.clip != 0.0):
            
            scaler.unscale_(optimizer) #To avoid underflow
            print([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None])
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

            # Compute gradient norms after clipping
            total_norm_after = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )
            
            if(step !=0):
                print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
        
        # Compute gradient norms for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         print(f"Gradient norm for {name}: {grad_norm}")
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        # torch.cuda.empty_cache()
        # optimizer.step()
        scheduler.step()
        # torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        # accumulated_loss = loss
        
        # torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        # loss /= world_size
        perplexity = torch.exp(torch.tensor(accumulated_loss.item()))  # Calculate perplexity
        # if(device == 0):
        wandb.log({
                    "Learning Rate": scheduler.get_last_lr()[0],
                    "Train_Loss": accumulated_loss.item(),
                    # "Train loss": loss.item(),
                    "Train Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": total_norm_before.item(),
                    # "Epoch": epoch
                    
        })


        if step % 200 == 0:
            count = 1
            while(count):  
                prompt = "Once upon a time "
                generated_text = topk_sampling(model, prompt, max_length=ModelArgs.block_size, top_k=50, temperature=1.0, device=device)
    
     
                print(f" Step: {step} | Generated Text: {generated_text}")

                save_to_file(step, generated_text)
                count -= 1
        
    

        # break
        # if step % 5 == 0:
        #     torch.cuda.empty_cache()
    # Cleanup
    # if device == 0:
        # writer.close()
    wandb.finish()
    cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()



