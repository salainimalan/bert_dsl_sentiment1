
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

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
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  
        x = self.dropout(pooled_output)
        return self.classifier(x).squeeze(1)


df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df = df[[0, 5]]
df.columns = ['label', 'text']
df = df[df['label'].isin([0, 4])]
df['label'] = df['label'].map({0: 0, 4: 1})
df = df.sample(50000, random_state=42).reset_index(drop=True)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TwitterDataset(train_texts, train_labels, tokenizer)
val_dataset = TwitterDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            pred_labels = (probs > 0.5).long().cpu().numpy()
            true_labels.extend(labels.long().cpu().numpy())
            preds.extend(pred_labels)
    acc = accuracy_score(true_labels, preds)
    return total_loss / len(loader), acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 3
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")

torch.save(model.state_dict(), "bert_only_model.pth")
print("\nModel saved to 'bert_only_model.pth' ")
