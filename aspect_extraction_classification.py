import os
import pandas as pd
from torch import nn
import re
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as functional
from transformers import BertTokenizer, BertModel
import nltk
from nltk.tokenize import word_tokenize

# Download models necessary for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define device and leverage GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the IMDB dataset function
def load_imdb_data(data_dir, split):
    data={'review':[], 'label':[]}

    for sentiment in ['pos', 'neg']:
        count = 0
        path = os.path.join(data_dir, split, sentiment)

        for file_name in os.listdir(path):
            # Set the number of dataset to be loadded
            # Can be adjusted to reduce duration of the Aspect Extraction process
            # Total dataset 2*set number with equal +/-
            if count >= 500: # Change this number
                break
            if file_name.endswith('.txt'):
                with open(os.path.join(path, file_name), 'r', encoding='utf-8') as file:
                    review = file.read()
                    data['review'].append(review)
                    data['label'].append(1 if sentiment == 'pos' else 0)
                    count +=1
    
    return pd.DataFrame(data)


# ~~~~~~~~~~~~~~~~~~~~~~~~~Aspect Extraction Process~~~~~~~~~~~~~~~~~~~~~~~~~
# Preprocess text function
def text_preprocessing(text):

    # Clean text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase

    # Tokenize text to extract the Part-of-speech
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)

    noun_phrases = []
    current_phrase = []

    for word, pos in pos_tags:

        # Extract the aspect word
        # Consider Noun, Singular Noun, Proper Noun, Proper Singular Noun
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            current_phrase.append(word)
        else:
            if current_phrase:
                noun_phrases.append(' '.join(current_phrase))
                current_phrase = []
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))

    return text, noun_phrases

# Load the IMDB dataset
data_dir='aclImdb' # Dataset folder directory
print('Loading only 1000 rows of train dataset')
train_df = load_imdb_data(data_dir, 'train')

print('Loading only 1000 rows of test dataset')
test_df = load_imdb_data(data_dir, 'test')


# Apply the text processing to the dataset
print("Preprocess text & Implement PoS tagging")
train_df['review'], train_df['aspects'] = zip(*train_df['review'].apply(text_preprocessing))
test_df['review'], test_df['aspects'] = zip(*test_df['review'].apply(text_preprocessing))

# Reduce the max length and batch size to reduce the time required
max_length = 256 # Define the max context length for tokenization
batch_size = 16 # Define the number of batch


# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Function to extract related words to the aspect
# This function can take long time to finish
# It is expected to get the aspects from the dataset in >30 minutes
# Adjust the number of rows in the loading dataset function
def extract_related_words_to_aspects(row):
    review = row['review']
    aspect_terms = row['aspects']

    try:
        tokens = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
        with torch.no_grad():
            token_embeddings = bert_model(**tokens).last_hidden_state[0]
        
        tokens_words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

        # Get the mean of the aspect embedding
        # Since an aspect word can consists of multiple tokens (multiple embedding)
        def get_mean_embedding(phrase):
            phrase_tokens = tokenizer(phrase, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            phrase_embeddings = token_embeddings[[i for i, token_id in enumerate(tokens['input_ids'][0]) if token_id in phrase_tokens]]
            
            if phrase_embeddings.size(0) == 0:
                return torch.zeros(token_embeddings.size(1))
            return torch.mean(phrase_embeddings, dim=0)

        aspect_embeddings = {aspect: get_mean_embedding(aspect) for aspect in aspect_terms}
        aspect_embeddings_stack = torch.stack(list(aspect_embeddings.values()))

        related_words = {aspect: [] for aspect in aspect_terms}

        for idx, token_embedding in enumerate(token_embeddings):
            # Skip BERT special tokens for similarity calculation
            if tokens_words[idx] in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            # Calculate cosine similarity of the aspect and the entire sentence
            similarities = functional.cosine_similarity(token_embedding.unsqueeze(0), aspect_embeddings_stack)
            best_aspect_idx = similarities.argmax().item() # Pick the highest similarity tokens

            if similarities[best_aspect_idx].item() > 0:  # Only consider positive similarities
                best_aspect = list(aspect_embeddings.keys())[best_aspect_idx]
                related_words[best_aspect].append(tokens_words[idx])

        # Combine aspects
        combined_tokens = []
        for tokens in related_words.values():
            if combined_tokens:
                combined_tokens.append('[SEP]') # Add [SEP] token between aspects
            combined_tokens.extend(tokens)

        combined_string = ' '.join(combined_tokens)
        
        return combined_string
    except:
        return ' '

# Apply aspect extraction to the dataset
print('Extracting aspects from reviews train dataset')
train_df['extracted_aspects'] = train_df.apply(
    lambda row: extract_related_words_to_aspects(row), axis=1)
print('Extracting aspects from reviews test dataset')
test_df['extracted_aspects'] = test_df.apply(
    lambda row: extract_related_words_to_aspects(row), axis=1)



# ~~~~~~~~~~~~~~~~~~~~~~~~~Aspect Sentiment Classification Process~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to Encode texts into tokens
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
model = model.to(device)

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
