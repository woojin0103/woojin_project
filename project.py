import torch
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. 데이터 로딩
path = "reviews_test.csv"
df = pd.read_csv(path)  # utf-8 기본 인코딩
data_X = df['review'].tolist()
labels = df['rating'].tolist()

# 3. 토크나이징
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, padding="max_length", return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 4. 데이터 분할
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_masks, val_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 5. DataLoader 구성
batch_size = 8
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
val_data = TensorDataset(val_inputs, val_masks, torch.tensor(val_labels))

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# 6. 모델 및 학습 설정
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader)*epochs)

# 7. 학습 및 평가 루프
epoch_results = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
        b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]

        model.zero_grad()
        output = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss = output.loss
        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # 평가
    model.eval()


    def evaluate(dataloader):
        preds, true = [], []
        for batch in dataloader:
            b_input_ids, b_mask, b_labels = [x.to(device) for x in batch]
            with torch.no_grad():
                output = model(b_input_ids, attention_mask=b_mask)
            logits = output.logits
            preds += torch.argmax(logits, dim=1).cpu().tolist()
            true += b_labels.cpu().tolist()
        accuracy = np.sum(np.array(preds) == np.array(true)) / len(true)
        return accuracy


    train_acc = evaluate(train_dataloader)
    val_acc = evaluate(val_dataloader)
    epoch_results.append((avg_train_loss, train_acc, val_acc))

# 8. 결과 출력
for i, (loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(f"Epoch {i}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# 9. 모델 저장
save_path = "mobilebert-finetuned-reviews"
model.save_pretrained(save_path + '.pt')
# tokenizer.save_pretrained(save_path)
print("모델 저장 완료:", save_path)
