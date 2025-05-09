import mediapipe as mp
from PIL import Image, ImageOps
import numpy as np
import transformers
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import os
import gradio as gr
import argparse

cookies='Your Cookie'
expression_model_path='best_model.pt path'
age_model_path='swinv2_ages.pt path'
gender_model_path='swinv2_gender.pt path'

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
def detect_and_crop_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_np = np.array(image)
        results = face_detection.process(image_np)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            xmin = int(bbox.xmin * iw)
            ymin = int(bbox.ymin * ih)
            width = int(bbox.width * iw)
            height = int(bbox.height * ih)
            xmax = xmin + width
            ymax = ymin + height
            face = image.crop((xmin, ymin, xmax, ymax))
            return face
        else:
            return -1


expression_test_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
age_test_transform=Compose([
    Resize((256,256)),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
gender_test_transform=Compose([
    Resize((256,256)),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])



model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
expression_model = torch.load(expression_model_file, map_location=device)
age_model = torch.load(age_model_file, map_location=device)
gender_model = torch.load(gender_model_file, map_location=device)
expression_model.eval()
age_model.eval()
gender_model.eval()
expression_model.to(device)
age_model.to(device)
gender_model.to(device)
label = ['화난', '행복한', '혼란스러운', '슬픈']
gender = ['남자', '여자']

import suno
def llama_run(text):
    PROMPT = '''당신은 유용한 생성형 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.'''
    instruction = f"'{text}'에게 어울리는 음악 장르와 분위기만 제안해줘."
    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
        ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=350,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )

    return outputs[0]["generated_text"][len(prompt):]
def model_run(img):
    if img is None:
        return "이미지를 업로드하세요.", None

    try:
        gr.Info('이미지 분석 중!!!')
        with torch.no_grad():
            img = ImageOps.exif_transpose(img)
            cropped_img = detect_and_crop_face(img)
            if cropped_img == -1:
                return "얼굴을 찾을 수 없습니다.", None

            expression_img = expression_test_transform(cropped_img)
            age_img = age_test_transform(cropped_img)
            gender_img = gender_test_transform(cropped_img)

            expression_img = expression_img.unsqueeze(0).to(device)
            age_img = age_img.unsqueeze(0).to(device)
            gender_img = gender_img.unsqueeze(0).to(device)

            expression_pred = expression_model(expression_img)
            age_pred = age_model(age_img)
            gender_pred = gender_model(gender_img)

            expression_pred = label[expression_pred.argmax(1).detach().cpu().numpy().tolist()[0]]
            age_pred = age_pred.detach().cpu().item()
            gender_pred = gender[(torch.sigmoid(gender_pred) > 0.5).int().detach().cpu().numpy().tolist()[0][0]]

        result_text = f'{expression_pred} 표정을 한 {int(age_pred)}세 {gender_pred}'
        result_prompt = llama_run(result_text)
        gr.Info('추천 기반 음악 장르와 분위기를 입력해주세요!')
        return result_text, result_prompt

    except Exception as e:
        return f"오류가 발생했습니다: {e}", None
def generate_songs(style, add_prompt,result_output):
    gr.Info('노래 생성 중 입니다!!!')
    file_path=[]
    client=suno.Suno(cookie=COOKIE)
    songs=client.generate(
        prompt=f'{result_output}, {style}, {add_prompt}',is_custom=False,wait_audio=True
    )
    for song in songs:
        file_path.append(client.download(song=song))
    gr.Info('노래 생성 완료 되었습니다!!!')
    return file_path[0], file_path[1]

"""#Gradio"""

custom_css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
}
h1 {
    color: #1e90ff;
    text-align: center;
}
.label, .output_text, .output_audio {
    font-size: 16px;
    font-weight: bold;
    color: #555;
}

.gr-tabitem {
    background-color: #ffffff;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}
.gr-tabitem.active {
    background-color: #f1f1f1;
}
.center {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}
.gradio-container {
    background: transparent;
    background: url('https://images.unsplash.com/photo-1445375011782-2384686778a0?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D') no-repeat center center fixed;
    background-size: cover;
}
"""

description_html = """
<div style="text-align: center; margin: 20px 0;">
    <p style="font-size: 1.2em; color: #333; max-width: 800px; margin: 0 auto;">
        This interface analyzes images uploaded by users to predict gender, age, and emotions. It also creates songs with analysis data.
    </p>
    <p style="font-size: 1.2em; color: #333; max-width: 800px; margin: 10px auto;">
        Click the <strong>Upload Images</strong> button below to upload an image
    </p>
    <p style="font-size: 1.2em; color: #ff69b4; max-width: 800px; margin: 20px auto;">
        Get ready to embark on a musical journey with your face!
    </p>
</div>
"""

js = """
window.onload = function() {
    var h1 = document.createElement('h1');
    h1.id = 'title-text';
    h1.style.fontFamily = "'Dancing Script', cursive";
    h1.style.fontSize = '3em';
    h1.style.textShadow = '2px 2px #ff1493';
    h1.innerText = '\\uD83C\\uDFB5 Music Face \\uD83C\\uDFB5';

    var descriptionContainer = document.querySelector('.gradio-container > div');
    descriptionContainer.insertBefore(h1, descriptionContainer.firstChild);

    var text = h1.innerText;
    h1.innerText = '';

    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                h1.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }
}
"""
def reset_interface():
    return None, None, None, None, None

with gr.Blocks(css=custom_css,js=js,title='Music Face',analytics_enabled=True) as iface:
    gr.Markdown(description_html)
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Images", height=300, width=300)
    with gr.Tabs():
        with gr.TabItem("Analysis"):
            result_output = gr.Textbox(label='분석 결과')
            result_prompt_output = gr.Textbox(label='Llama3 기반 추천 결과')
            style_box = gr.Textbox(label="음악 장르", placeholder="Kpop")
            add_box = gr.Textbox(label='추가 프롬프트', placeholder="Female vocals")
        with gr.TabItem("Music1"):
            music1_output = gr.Audio(label='Music1')
        with gr.TabItem("Music2"):
            music2_output = gr.Audio(label='Music2')
    img_input.change(fn=model_run, inputs=img_input, outputs=[result_output, result_prompt_output])
    generate_button = gr.Button('음악 생성')
    clear = gr.Button('다른 이미지 시도')
    generate_button.click(fn=generate_songs, inputs=[style_box, add_box, result_output], outputs=[music1_output, music2_output])
    clear.click(fn=reset_interface, inputs=[], outputs=[img_input, result_output, result_prompt_output, music1_output, music2_output], queue=False)
iface.launch(show_error=True,share=True)

