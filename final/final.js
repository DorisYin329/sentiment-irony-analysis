const manualBtn =
document.getElementById("manualBtn");

const dcardBtn =
document.getElementById("dcardBtn");

const manualSection =
document.getElementById("manualSection");

const dcardSection =
document.getElementById("dcardSection");

const loading =
document.getElementById("loading");



// ======================
// 模式切換
// ======================

manualBtn.onclick = () => {

    manualBtn.classList.add("active");

    dcardBtn.classList.remove("active");

    manualSection.classList.remove("hidden");

    dcardSection.classList.add("hidden");

};



dcardBtn.onclick = () => {

    dcardBtn.classList.add("active");

    manualBtn.classList.remove("active");

    dcardSection.classList.remove("hidden");

    manualSection.classList.add("hidden");

};



// ======================
// 手動分析
// ======================

const analyzeBtn =
document.getElementById("analyzeBtn");


if(analyzeBtn){

    analyzeBtn.addEventListener(

        "click",

        async()=>{

            const text =
            document
            .getElementById("textInput")
            .value;


            if(text.trim()==""){

                alert("請輸入文字");

                return;
            }


            loading.style.display = "block";


            try{

                const response =
                await fetch(

                    "http://127.0.0.1:5000/predict",

                    {

                        method:"POST",

                        headers:{
                            "Content-Type":
                            "application/json"
                        },

                        body:JSON.stringify({
                            text:text
                        })

                    }

                );


                const data =
                await response.json();


                loading.style.display = "none";


                let color =
                data.sentiment;


                document
                .getElementById(
                    "manualResult"
                )

                .innerHTML =

                `

                <div class="card ${color}">

                    <h3>

                        原文：

                        「${text}」

                    </h3>


                    <div>

                        <span class="tag tag1">

                            ${data.sentiment}

                        </span>


                        <span class="tag tag2">

                            情緒：

                            ${data.emotion}

                        </span>


                        <span class="tag tag3">

                            行為：

                            ${data.dialogue_act}

                        </span>

                    </div>


                    <div class="analysis">

                        <h4>

                            💡 AI語意分析

                        </h4>

                        ${data.human_analysis}

                    </div>

                </div>

                `;

            }

            catch(error){

                console.log(error);

                alert("API連線失敗");

                loading.style.display = "none";

            }

        }

    );

}



// ======================
// PTT分析
// ======================

const pttBtn =document.getElementById("pttAnalyzeBtn");
const sentimentMap = {

    positive:"正向",

    negative:"負向",

    neutral:"中性"

};

const emotionMap = {

    happy:"快樂",

    sadness:"難過",

    anger:"生氣",

    relaxed:"放鬆",

    astonished:"驚訝",

    neutral:"平靜",

    depress:"低落"

};

if(pttBtn){

    pttBtn.onclick = async()=>{

        loading.style.display = "block";


        try{

            const response =
            await fetch(
                "http://127.0.0.1:5000/analyze_ptt"
            );

            const datas =
            await response.json();


            loading.style.display = "none";


            let html = "";


            datas.forEach(data=>{

                html += `

                <div class="card">

                    <h3>

                        📌 ${data.title}

                    </h3>

                `;


                data.analysis.forEach(comment=>{

                    html += `

                    <div class="commentCard">

                        <div class="previewText">

                            ${comment.text}

                        </div>


                        <div style="margin-top:15px;">

                            <span class="tag tag1">

                                ${sentimentMap[comment.sentiment] || comment.sentiment}

                            </span>


                            <span class="tag tag2">

                                ${emotionMap[comment.emotion] || comment.emotion}

                            </span>


                            <span class="tag tag3">

                                ${comment.dialogue_act}

                            </span>

                        </div>

                    </div>

                    `;

                });


                html += `

                </div>

                `;

            });


            document
            .getElementById(
                "pttResult"
            )

            .innerHTML = html;

        }

        catch(error){

            console.log(error);

            alert("PTT分析失敗");

            loading.style.display = "none";

        }

    };

}