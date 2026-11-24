import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW 
from torch import nn

# 2. 定義資料集類別：加入上下文抓取邏輯
class CPEDDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 記住：行是 column，列是 row。
        # 從 DataFrame 提取當前句
        current_text = str(self.df.iloc[idx]['Utterance_TW'])
        
        # --- 上下文處理邏輯 ---
        prev_text = ""
        # 如果不是第一筆資料，且當前句與前一句屬於同一個對話 ID
        if idx > 0 and self.df.iloc[idx]['Dialogue_ID'] == self.df.iloc[idx-1]['Dialogue_ID']:
            prev_text = str(self.df.iloc[idx-1]['Utterance_TW'])

        # BERT Tokenizer 傳入兩個字串，自動生成 [SEP] 分隔的雙句結構
        encoding = self.tokenizer(
            prev_text,        # Sentence A (上下文)
            current_text,     # Sentence B (當前目標句)
            truncation=True, 
            padding='max_length',
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sent': torch.tensor(self.df.iloc[idx]['sentiment_label'], dtype=torch.long),
            'emo': torch.tensor(self.df.iloc[idx]['emotion_label'], dtype=torch.long),
            'da': torch.tensor(self.df.iloc[idx]['da_label'], dtype=torch.long)
        }

# 3. 多任務 BERT 模型 (維持 da_classifier 為 19 類)
class MultiTaskBERT(nn.Module):
    def __init__(self):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.sent_classifier = nn.Linear(768, 3) 
        self.emo_classifier = nn.Linear(768, 13)
        self.da_classifier = nn.Linear(768, 19)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.sent_classifier(pooled_output), \
               self.emo_classifier(pooled_output), \
               self.da_classifier(pooled_output)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用裝置: {device} | 啟動：雙句語境強化特訓！")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 使用已經修正過反諷標籤的資料集
    dataset = CPEDDataset('train_context_ultra_fix.csv', tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiTaskBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-6) 

    # 針對 Irony (ID 7) 進行 8 倍加權，維持妳之前的調優成果
    da_weights = torch.ones(19).to(device)
    da_weights[7] = 8.0 
    
    loss_fn_s = nn.CrossEntropyLoss()
    loss_fn_e = nn.CrossEntropyLoss()
    loss_fn_d = nn.CrossEntropyLoss(weight=da_weights)

    num_epochs = 2 
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\n 第 {epoch+1} 回合上下文特訓開始...")
        
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets_s = batch['sent'].to(device)
            targets_e = batch['emo'].to(device)
            targets_d = batch['da'].to(device)
            
            optimizer.zero_grad()
            out_s, out_e, out_d = model(input_ids, mask)
            
            # 情緒權重提升 1.5 倍
            l_s = loss_fn_s(out_s, targets_s)
            l_e = loss_fn_e(out_e, targets_e) * 1.5 
            l_d = loss_fn_d(out_d, targets_d)
            
            loss = l_s + l_e + l_d
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}", end='\r')
        
        # 存檔名稱加上 context 標記
        save_path = f'context_irony_model_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"\n Epoch {epoch+1} 完成！平均 Loss: {total_loss/len(loader):.4f}")

    print("\n 訓練結束！")