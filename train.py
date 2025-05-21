# !pip install torchtune
# !pip install torchao
# !pip install wandb


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


# Load model directly
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", token='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')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#liger kernels
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss




@dataclass
class ModelArgs:
    #Hyperparameters
    #Inspired by SmolLM 133 M by HuggingFace
    block_size = 1024 
    batch_size = 32
    embeddings_dims = 512
    attn_dropout = 0.1
    no_of_heads = 8 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 3
    max_lr = 2.5e-4
    no_of_decoder_layers = 8 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    device = 'cuda:5'
    no_kv_heads = 2
    scaling_factor = 0.5
    vocab_size = len(tokenizer.get_vocab()) + 768
    local_block_size = 256
    base_freq=10000
    clip = 1.0
    use_liger = True  # Flag to control whether to use LigerFusedLinearCrossEntropyLoss
    inference = False  # Flag to indicate inference mode
    eps = 1e-8  # For AdamW optimizer
    
#Datasets

# Using tinyshakespeare

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



#Subword level tokenization

#Loading custom trained BPE
# Load the tokenizer
# tokenizer = Tokenizer.from_file("data/bpe_tokenizer_tinyshakespeare_1k.json")
# vocab_size = tokenizer.get_vocab_size()
# Encode and decode functions
# encode = lambda s: tokenizer.encode(s).ids
# decode = lambda l: tokenizer.decode(l)





###############################################################################
#Character level tokenization

# # here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# create a mapping from characters to integers
stoi = { ch: i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - ModelArgs.block_size, (ModelArgs.batch_size,))
    x = torch.stack([data[i:i+ModelArgs.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ModelArgs.block_size+1] for i in ix])
    x, y = x.to(ModelArgs.device), y.to(ModelArgs.device)
    return x, y


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
    print(fw_train)




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)
 
    def collate_fn(batch):
        # Extract text data
        texts = [item ["text"] for item in batch]

        input_encodings = tokenizer(texts, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        
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
            shuffle=False
        )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_test,
              
            
            batch_size=batch_size,
            # sampler=DistributedSampler(fw_test, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False
        )
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            
            
            # sampler=DistributedSampler(fw_train['train'], shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
                # generator=generator,
            # sampler=DistributedSampler(fw_train["test"]),
            collate_fn=collate_fn,
              
            drop_last=True,
            shuffle=True
        )
    return data_loader





    
    

# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    
    for _ in range(max_length):
        with torch.no_grad(), torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
            # Pass inference=True to use the inference path in the model
            outputs = model(input_ids, inference=True)
            logits = outputs[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)



class Normalization(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):  
        super().__init__()
        self.rmsnorm_layer = RMSNorm(embeddings_dims)
        
        
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
        self.hidden_dims = int(2 * ( 4 * embeddings_dims) / 3)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




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

        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        # self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        x = self.linear_layer(x)
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




class Gemma(nn.Module):
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
        
        # Initialize LigerFusedLinearCrossEntropyLoss for optimized training
        if ModelArgs.use_liger:
            self.le_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id
            ).to(device)
        
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
               
                     
                    
    def forward(self, x, actual_labels=None, inference=False):
        global_base_freq = 100000 
        local_base_freq = 10000
        index = 0
        no_of_layers = 0
        x = self.embeddings(x)
        x = self.dropout(x)
        
        for layer in self.decoder:
            if no_of_layers % 5 == 0:
                x = layer(x, global_base_freq)
            else:
                # Create a mask with 1's in the local block and 0's elsewhere
                batch_size, seq_len, emb_dim = x.shape
                local_end = min(index + ModelArgs.local_block_size, seq_len)
                
                # Create a mask of zeros, then fill the local attention window with ones
                mask = torch.zeros(batch_size, seq_len, 1, device=x.device)
                mask[:, :local_end, :] = 1.0
                reverse_mask = torch.ones(batch_size, seq_len, 1, device=x.device)
                reverse_mask[:, :local_end, :] = 0.0
                
                # Apply the mask and process only the active region
                rest_input_vals = x * reverse_mask
                local_block = x * mask
                processed_local = layer(local_block, local_base_freq)
                x = processed_local + rest_input_vals
                
                # Move index forward for next local window
                index += ModelArgs.local_block_size
            
            no_of_layers += 1
            
        x = self.norm(x)
        
        # Handle different modes (inference vs training) and loss types
        if inference:
            out = self.linear_layer(x)
            return out
            
        if ModelArgs.use_liger and actual_labels is not None:
            # Use the optimized LigerFusedLinearCrossEntropyLoss for training
            y = x.contiguous().view(-1, ModelArgs.embeddings_dims)
            labels = actual_labels.contiguous().view(-1)
            
            # Pass linear layer weights FIRST as required by Liger
            loss = self.le_loss(self.linear_layer.weight, y, labels)
            return loss
        else:
            # Standard forward pass with linear layer
            out = self.linear_layer(x)
            return out
    


# Instantiating the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# ModelArgs.device = device
model = Gemma(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=ModelArgs.device)
model = model.to(ModelArgs.device)

# model = DDP(model, device_ids=[gpu_ids])


#Printing a summary of the architecture
from torchinfo import summary
idx, targets = get_batch('test')
idx = idx.to(ModelArgs.device)
print(summary(model=model,
        input_data=idx,
        # input_size=(ModelArgs.batch_size, ModelArgs.block_size, ModelArgs.embeddings_dims),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))


def save_text(file_path, step, text):
    with open(file_path, 'w') as f:
        f.write(f"Step {step}: {text}\n")
        
        


save_checkpoint_iter = 2000
total_iters = 20000
eval_iters = 200
eval_check = 200
warmup_iters = 700
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 20000
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * 1))



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


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused


# import tqdm 
def train():
    # Set device to CUDA if available
    device = ModelArgs.device
    print(f"Start running training on {device}.")
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project = 'Gemma-Training',
        # config = ModelArgs, # you can uncomment this to log model config
    )
    
    # Create model and move to GPU
    model = Gemma(embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, 
                  vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout, device=device)
    model = model.to(device)

    print("Model loaded")
    # Setup optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr)
    
    # Training parameters
    # save_checkpoint_iter = 2000
    # total_iters = 610000
    # eval_iters = 1000

    
    # Training progress bar
    train_epoch_iterator = tqdm.tqdm(range(total_iters), desc="Training")
    val_dataloader = prepare_dataset('val', device, ModelArgs.batch_size)
    val_iterator = iter(val_dataloader)
    # Get batches for training
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        count = 0
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):

                nonlocal val_iterator
                
                # for k, batch in enumerate(dataloader):
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    batch = next(val_iterator)

                
                idx = batch['input_ids'].to(device)
        
                targets = batch['labels'].to(device)
                
                if ModelArgs.use_liger:
                    # Pass actual labels to the model to use optimized loss function
                    loss = model(idx, actual_labels=targets)
                else:
                    # Standard cross entropy path
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    
                    targets = targets.view(batch_size * block_size)
                    
                    loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                
                losses[k] = loss.item()
                # count += 1
            out[split] = losses.mean()

        model.train()
        return out
    token_count = 0
    # Start training loop
    model.train()
    print("Lessgoo...")
    # print("gradient steps: ", gradient_accumulation_steps)
    dataloader = prepare_dataset('train', device, ModelArgs.batch_size)
    train_dataloader = iter(dataloader) 
    accumulated_loss = 0.0
    
    for epoch in range(ModelArgs.epochs):
        for step in train_epoch_iterator:
            # Periodically evaluate loss on train and val sets
            if (step % eval_iters == 0 and step != 0) or step == total_iters - 1:
                losses = estimate_loss()
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                print(f"step {step}: train loss {accumulated_loss:.4f}, val loss {losses['val']:.4f}")
                val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                # Log metrics to wandb
                wandb.log({
                    "val_perplexity": val_perplexity,
                    # "val_step_loss": losses['train'],
                    "val_step_loss": losses['val'],
                    "step": step
                })
            
            # Save checkpoint periodically
            if step % save_checkpoint_iter == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                torch.save(model.state_dict(), f"checkpoint_{step}.pt")
                print("Checkpoint saved")
            
            # Get batch for training step
            try:
                batch = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(dataloader)
                batch = next(train_dataloader)
                
            # # for batch in dataloader:
            # input_ids = batch["input_ids"].to(device)
            # targets = batch["labels"].to(device)
            accumulated_loss = 0.0  
            for micro_step in range(gradient_accumulation_steps):
                
                try:
                    batch = next(train_dataloader)
                except StopIteration:
                    train_dataloader = iter(dataloader)
                    batch = next(train_dataloader)
                    
                    
                idx = batch['input_ids'].to(device)
        
                targets = batch['labels'].to(device)
                
                token_count += len(idx) * ModelArgs.batch_size
                
                
                # with torch.autocast(device_type=ModelArgs.device, dtype=torch.bfloat16):
                # Use LigerFusedLinearCrossEntropyLoss for efficient training
                if ModelArgs.use_liger:
                    # Pass actual labels to the model to use optimized loss function
                    loss = model(idx, actual_labels=targets)
                else:
                    # Standard cross entropy path
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    
                    targets = targets.view(batch_size * block_size)
                    
                    loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                
                loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
                accumulated_loss += loss.detach()
                loss.backward()
                    
                
            # break
                if(device == 0):
                    if(micro_step % 10 == 0):
                #     if(step == train_loader_length):
                #       break
                        
                        print("Micro Batch : ", micro_step)
                        print("Step : ", step, "/", total_iters)
                        print('Total batches: ', len(train_dataloader))
                        print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                        print("Total tokens processed: ", token_count)
                # count += 1
            
            
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
                
            unused_params = find_unused_parameters(model)
            if unused_params:
                    print(f"Unused parameters: {unused_params}")
            
            # Compute gradient norms before clipping
            if(ModelArgs.clip != 0.0):
                # scaler.unscale_(optimizer) #To avoid underflow
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)

                # Compute gradient norms after clipping
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                
                if(device  == 0 and step !=0):
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            optimizer.step()
            # accumulated_loss = loss.item()
            # accumulated_loss /= gradient_accumulation_steps
            perplexity = torch.exp(torch.tensor(accumulated_loss)).item()  # Calculate perplexity
            # if(device == 0):
            wandb.log({
                        "Learning Rate": optimizer.param_groups[0]['lr'],
                        "Train_Loss": accumulated_loss,
                        # "Train loss": loss.item(),
                        "Train Perplexity": perplexity,
                        "Total Tokens Processed": token_count,
                        "Step": step,
                        "Gradient Norm": total_norm_before.item(),
                        # "Epoch": epoch
                        
            })
            
            if(step !=0 and step % eval_iters == 0):
                    prompt = "Once upon a time in a land far, far away, "
                    generated_text = topk_sampling(model, prompt, max_length=ModelArgs.block_size, top_k=50, temperature=1.0, device=device)
        
        
                    print(f" Step: {step} | Generated Text: {generated_text}")
                    save_text(f"generated_data/generated_text_{step}.txt", step, generated_text)
        # Finish wandb run
        wandb.finish()

# Print CUDA device count but won't be using DDP
world_size = torch.cuda.device_count()
print(f"CUDA devices available: {world_size}")
train()