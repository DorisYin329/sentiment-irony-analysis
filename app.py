from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from analyzer import DcardEmotionAnalyzer

app = Flask(__name__)
CORS(app)

print("載入模型中...")

analyzer = DcardEmotionAnalyzer(
    model_path="context_irony_model_epoch5.pth",
    mapping_path="multitask_mappings.json"
)

print("模型載入完成")


# 手動輸入分析
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    text = data["text"]

    result = analyzer.predict(
        "",
        text
    )

    return jsonify(result)


# PTT CSV分析
@app.route("/analyze_ptt")
def analyze_ptt():

    df = pd.read_csv(
        "ptt_boygirl_articles.csv"
    )

    df = df[
        df["title"].str.contains(
        "交男女朋友要有顏值和錢是基本"
        )
    ]

    results = []

    for i in range(len(df)):

        title = str(df.iloc[i]["title"])

        comments = str(df.iloc[i]["comments"])

        comment_list = comments.split("|")

        merged_comments = []

        temp = ""

        last_user = ""

        for c in comment_list:        

            c = c.strip()

            if len(c) < 3:
                continue

            match = re.match(r"(推|噓|→)\s*([^:：]+)",c)

            if match:
                
                user = match.group(2)

                # 同一作者 → 接在一起
                if user == last_user:

                    parts = re.split(r"[:：]",c,maxsplit=1)

                    if len(parts) > 1:

                        clean_text = parts[1].strip()
                    else:

                        clean_text = c

                    temp += " " + clean_text

                # 新作者
                else:

                    if temp != "":
                        merged_comments.append(temp)

                    temp = c

                    last_user = user

            # 沒有作者 → 視為上一句續行
            else:

                temp += " " + c

        # 最後一筆加入
        if temp != "":
            merged_comments.append(temp)

        
        single_results = []

        for c in merged_comments[:8]:

            c = c.strip()

            if len(c) < 3:
                continue

            result = analyzer.predict(
                "",
                c
            )

            single_results.append({

                "text": c,

                "sentiment":
                result["sentiment"],

                "emotion":
                result["emotion"],

                "dialogue_act":
                result["dialogue_act"]

            })

        results.append({

            "title": title,

            "analysis": single_results,

        })

    return jsonify(results)


@app.route("/")
def home():

    return "API成功啟動"


if __name__ == "__main__":

    app.run(debug=True)