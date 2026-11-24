import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import json

# 1. 模型架構 (必須放在這裡才能載入權重)
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
        return self.sent_classifier(pooled_output), self.emo_classifier(pooled_output), self.da_classifier(pooled_output)


# 2. 分析器主體 (給後端呼叫的介面)
class DcardEmotionAnalyzer:
    def __init__(self, model_path='context_irony_model_epoch5.pth', mapping_path='multitask_mappings.json'):
        # 自動偵測環境是否有 GPU，沒有的話會自動退回 CPU 模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 載入模型
        self.model = MultiTaskBERT().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 載入標籤
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)

    # 隱藏在內部的語意合成器 (完整版)
    def _generate_human_text(self, s_name, e_name, d_name):
        # 建立繁體中文翻譯字典
        sentiment_tw = {"positive": "正向", "neutral": "平淡", "negative": "負面"}
        
        emotion_tw = {
            "happy": "快樂", "grateful": "感激", "relaxed": "輕鬆", "other-positive": "些許愉悅",
            "neutral": "平靜", "angry": "憤怒", "anger": "憤怒", "sad": "難過", "sadness": "難過",
            "feared": "恐懼", "fear": "恐懼", "depressed": "沮喪", "depress": "沮喪",
            "disgusted": "厭惡", "disgust": "厭惡", "astonished": "驚訝", "worried": "擔憂", 
            "negative-other": "其他負面"
        }

        da_tw = {
            "greeting": "問候", "question": "提問", "answer": "回答", 
            "statement-opinion": "主觀評論", "statement-non-opinion": "客觀陳述", 
            "apology": "道歉", "command": "指令", "c": "要求", "advice": "建議",
            "agreement": "同意", "disagreement": "反駁", "dag": "異議",
            "acknowledge": "接收訊息", "a": "回應", "appreciation": "讚賞", 
            "interjection": "感嘆", "conventional-closing": "結束語", "fc": "客套收尾",
            "thanking": "感謝", "quotation": "引用", "reject": "拒絕", "rj": "拒絕",
            "irony": "反諷", "comfort": "安慰", "other": "其他行為", "oth": "其他"
        }

        s_tw = sentiment_tw.get(s_name, s_name)
        e_tw = emotion_tw.get(e_name, e_name)
        d_tw = da_tw.get(d_name, d_name)

        # --- 強化後的 6 大情境診斷邏輯 ---
        
        # 1. 核心亮點：抓出陰陽怪氣 (Irony)
        if d_name == "irony":
            if s_name == "positive":
                return f"語意解析：這是一句誇張的「{e_tw}」式稱讚，帶點幽默感（語境：{s_tw}）。"
            return f"語意解析：這句話帶有「{e_tw}」的反諷語氣，聽起來有點酸喔（語境：{s_tw}）。"

        # 2. 處理社交暖心行為 (感謝、安慰、道歉)
        elif d_name in ["comfort", "thanking", "apology", "appreciation"]:
            if d_name == "apology":
                return f"語意解析：對方正在誠懇地「{d_tw}」，流露出「{e_tw}」的情緒（語境：{s_tw}）。"
            return f"語意解析：感受到濃濃的「{e_tw}」與「{d_tw}」，這是很溫暖的互動！（語境：{s_tw}）。"

        # 3. 處理強烈負面情緒
        elif s_name == "negative" and e_name not in ["neutral"]:
            return f"語意解析：這是一次情緒較強烈的「{d_tw}」，充滿了「{e_tw}」的氛圍（語境：{s_tw}）。"

        # 4. 處理一般互動 (提問、回答、陳述、驚訝)
        else:
            if e_name == "astonished":
                return f"語意解析：這句話表現出極度的「{e_tw}」，看來發生了意料之外的事！（語境：{s_tw}）。"
            
            # 針對模型最愛給的「主觀評論」與「客觀陳述」做特化潤飾
            if d_name in ["statement-opinion", "statement-non-opinion"]:
                if s_name == "positive":
                    return f"語意解析：對方發表了帶有「{e_tw}」情緒的{d_tw}，整體感覺相當積極、正向（語境：{s_tw}）。"
                elif s_name == "neutral":
                    return f"語意解析：這是一句語氣「{e_tw}」的{d_tw}，屬於中性的資訊表達（語境：{s_tw}）。"
                else:
                    return f"語意解析：這是一次情緒較為「{e_tw}」的{d_tw}，帶有負面的氛圍（語境：{s_tw}）。"
            
            # 針對「提問」與「回答」做特化
            if d_name in ["question", "answer"]:
                return f"語意解析：這是一次帶有「{e_tw}」語氣的{d_tw}，雙方正在進行資訊交流（語境：{s_tw}）。"

            # 最底層的保底
            return f"語意解析：這是一句表現出「{e_tw}」的{d_tw}，屬於一般的對話往來（語境：{s_tw}）。"

    # 給後端呼叫的唯一方法
    def predict(self, prev_text, curr_text):
        # 將文字轉為模型可接受的 Tensor 格式
        inputs = self.tokenizer(prev_text, curr_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length').to(self.device)
        
        # 進行推論 (不計算梯度以節省記憶體)
        with torch.no_grad():
            out_s, out_e, out_d = self.model(inputs['input_ids'], inputs['attention_mask'])
            
            # 取得預測結果的 Index，並轉換為文字標籤
            s_name = self.mappings['sentiment']['to_name'][str(torch.argmax(out_s, dim=1).item())]
            e_name = self.mappings['emotion']['to_name'][str(torch.argmax(out_e, dim=1).item())]
            d_name = self.mappings['da']['to_name'][str(torch.argmax(out_d, dim=1).item())]

            # 呼叫內部的語意合成器產出白話文
            human_text = self._generate_human_text(s_name, e_name, d_name)

            # 回傳一個乾淨的 JSON 相容字典給後端
            return {
                "sentiment": s_name,
                "emotion": e_name,
                "dialogue_act": d_name,
                "human_analysis": human_text
            }