import torch
import matplotlib.pyplot as plt
import re
import torch.nn as nn
from transformers import BertModel

# Define your custom model again
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        batch_size, seq_len, hidden_size = hidden_states.size()
        third = seq_len // 3
        start_emb = hidden_states[:, :third, :].mean(dim=1)
        end_emb = hidden_states[:, -third:, :].mean(dim=1)
        delta = end_emb - start_emb

        out = self.classifier(self.dropout(delta))
        return out.squeeze(1)
# Simple sentence splitter (fallback from NLTK)
def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s for s in sentences if s]

# Group sentences into N segments (based on paragraph length)
def group_sentences(sentences, num_segments=3):
    total = len(sentences)
    seg_size = max(1, total // num_segments)
    grouped = [' '.join(sentences[i:i+seg_size]) for i in range(0, total, seg_size)]
    return grouped

# Tokenize and get embedding for one chunk
def get_chunk_embedding(text, model, tokenizer, device, max_len=64):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_len
    ).to(device)
    with torch.no_grad():
        outputs = model.bert(**encoding)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding

# Score embedding
def get_sentiment_score(embedding, classifier_head):
    with torch.no_grad():
        score = classifier_head(embedding)
        return torch.sigmoid(score).item()

# Final sentiment shift visualizer
def plot_sentiment_shifts(paragraph, model, tokenizer, device, num_segments=3):
    sentences = split_text_into_sentences(paragraph)
    text_segments = group_sentences(sentences, num_segments=num_segments)

    scores = []
    for seg in text_segments:
        emb = get_chunk_embedding(seg, model, tokenizer, device)
        score = get_sentiment_score(emb, model.classifier)
        print(f"Segment: \"{seg.strip()}\" → Sentiment: {score:.2f}")
        scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(scores)+1), scores, marker='o', color='blue')
    plt.title("Sentiment Shift Across Paragraph")
    plt.xlabel("Segment")
    plt.ylabel("Sentiment Score (0 = Negative, 1 = Positive)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_DSL_Model()
model.load_state_dict(torch.load("bert_dsl_model.pth", map_location=device))
model.to(device)
model.eval()

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Your paragraph
paragraph = """I was really looking forward to this movie. The trailer looked amazing, and the cast seemed perfect for the story. The first few scenes were captivating and set a great tone. I especially loved the cinematography and the background score in the beginning. However, as the plot progressed, things started to feel a bit rushed. The character development felt shallow, and some plot points were confusing. By the time we reached the climax, I was already losing interest. The final twist was predictable and didn't have the emotional impact it was aiming for. Still, I appreciated the performances of a few actors, especially the lead, who did their best with the weak script. Overall, it was a mixed experience—great potential, but poorly executed."""

# Run visualization
plot_sentiment_shifts(paragraph, model, tokenizer, device, num_segments=3)