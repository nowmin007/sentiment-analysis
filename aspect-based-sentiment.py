import os
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

# Define device to leverage GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-uncased' # Define the model name
tokenizer = BertTokenizer.from_pretrained(model_name) # Define the toknizer to use
bert_model = BertModel.from_pretrained(model_name) # Load the model

# Load the preprocessed train and test dataset
train_df = pd.read_pickle('train_df.pkl')
test_df = pd.read_pickle('test_df.pkl')

# Reduce the batch size and max length if resource limited -> Might give different result
max_length = 256 # Set the maximum context length for the tokenizer
batch_size = 16  # Set the batch size to run simultaneously

def bert_encode(text):
    tokens = tokenizer.batch_encode_plus(text,
                                         return_tensors='pt',
                                         padding=True,
                                         truncation=True,
                                         max_length=max_length,
                                         add_special_tokens=True)
    return tokens['input_ids'][0], tokens['attention_mask'][0], tokens['token_type_ids'][0]


# Prepare the dataframe into Pytorch Dataset
class IMDBDataset(Dataset):
    def __init__(self, dataframe):
        self.extracted_aspects = dataframe['extracted_aspects'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.extracted_aspects)
    def __getitem__(self, idx):
        extracted_aspects = self.extracted_aspects[idx]
        label = self.labels[idx]
        input_ids, attention_mask, token_type_ids = bert_encode([extracted_aspects])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }
    

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad sentences to same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }

# Prepare PyTorch DataLoader
train_encoded = IMDBDataset(train_df)
test_encoded = IMDBDataset(test_df)

train_dataset = DataLoader(train_encoded, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataset = DataLoader(test_encoded, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Positional embedding class
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, d_model):
        super(PositionalEmbedding, self).__init__()
        self.max_length = max_length
        self.positional_embed = torch.zeros(max_length, d_model)
        self.position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.positional_embed[:,0::2] = torch.sin(self.position * self.div_term)
        self.positional_embed[:,1::2] = torch.cos(self.position * self.div_term)
        self.positional_embed = self.positional_embed.unsqueeze(0)

    def forward(self, x):
        x = x + self.positional_embed[:, :x.size(1)].to(device)
        return x


# Class of sentiment classifier model
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, dim_feedforward, output_dim, dropout, n_head, num_transformer_layers):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.positional_embedding = PositionalEmbedding(max_length, bert_model.config.hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(bert_model.config.hidden_size,
                                                        nhead=n_head,
                                                        batch_first=True,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=num_transformer_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bert_model.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        embedding_output = self.positional_embedding(hidden_states)
        transformer_output = self.transformer_encoder(embedding_output)
        hidden = self.dropout(transformer_output[:,0,:]) # Use the output corresponding to [CLS] token
        output = self.fc(hidden)
        return output
    
    
dim_feedforward = 256 # Dimension of the feedforward network
output_dim = 1 # Number of output classes
dropout = 0.3 # Number od dropout
n_head = 8 # Number of multi-head attention
num_transformer_layers = 2 # Number of transformer layers

model = SentimentClassifier(bert_model, dim_feedforward, output_dim, dropout, n_head, num_transformer_layers)
model = model.to(device) # Move the model to device

# Training parameters
epochs = 3
learning_rate=2e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Model training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    num_total_loss = 0
    num_corr_pred = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        num_total_loss += loss.item()
        predictions = torch.round(torch.sigmoid(outputs.squeeze()))
        num_corr_pred += torch.sum(predictions == labels).item()

    return num_total_loss / len(dataloader), num_corr_pred / len(dataloader.dataset)


# Model evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    num_total_loss = 0
    num_corr_pred = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device, dtype=torch.float)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs.squeeze(), labels)
            num_total_loss += loss.item()
            predictions = torch.round(torch.sigmoid(outputs.squeeze()))
            num_corr_pred += torch.sum(predictions == labels).item()

    return num_total_loss / len(dataloader), num_corr_pred / len(dataloader.dataset)


# Execute train and evaluation based on the number of epochs
print("Training Model")
for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_dataset, optimizer, criterion, device)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    val_loss, val_acc = evaluate_model(model, test_dataset, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
