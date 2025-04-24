
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class BERT_DSL_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state 

        batch_size, seq_len, hidden_size = hidden_states.size()
        third = seq_len // 3
        start_emb = hidden_states[:, :third, :].mean(dim=1)
        end_emb = hidden_states[:, -third:, :].mean(dim=1)
        delta = end_emb - start_emb
        out = self.classifier(self.dropout(delta))
        return out.squeeze(1)


df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['label', 'text']
df = df[df['label'].isin([0, 4])]
df['label'] = df['label'].map({0: 0, 4: 1})
df = df.sample(50000, random_state=42).reset_index(drop=True)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = TwitterDataset(train_texts, train_labels, tokenizer)
val_dataset = TwitterDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    from tqdm import tqdm
    for batch in tqdm(dataloader, desc='Training', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        from tqdm import tqdm
    for batch in tqdm(dataloader, desc='Training', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().cpu().numpy()
            labels = labels.long().cpu().numpy()

            predictions.extend(preds)
            targets.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = BERT_DSL_Model().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 3
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")


torch.save(model.state_dict(), "bert_dsl_model.pth")
print("Model saved to bert_dsl_model.pth ")

