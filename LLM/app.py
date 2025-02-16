# app.py
# -*- coding: utf-8 -*-

import os
import re
import base64
import gradio as gr
import wandb
from types import SimpleNamespace

# Import your local modules
from chain import get_answer, load_chain, load_vector_store
from config import default_config

# Remove diacritics from Arabic text (if needed)
def remove_diacritics(text):
    diacritic_pattern = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
    return re.sub(diacritic_pattern, '', text)

# Configure OpenAI API Key
def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open('C:/dev/EE569/Assignment2-LLM/LLM/key.txt', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    print("OpenAI API key configured")

configure_openai_api_key()

# For logging with Weights & Biases
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "llmapps"

# Convert image to Base64 so it can be embedded directly in the HTML
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        b64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"

# Preload the image
university_logo = get_base64_image(r'C:\dev\EE569\Assignment2-LLM\LLM\static\university_logo.png')

class Chat:
    """
    A chatbot interface that persists the vectorstore and chain between calls.
    """
    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.chain = None
  
    def __call__(self, question: str, history=None, openai_api_key=None):
        """
        This method is called whenever a user submits a question.
        """
        if openai_api_key:
            openai_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("الرجاء إدخال مفتاح OpenAI API الخاص بك")

        # Load the vector store once
        if self.vector_store is None:
            self.vector_store = load_vector_store(self.wandb_run, openai_key)

        # Load the chain once
        if self.chain is None:
            self.chain = load_chain(self.wandb_run, self.vector_store, openai_key)

        history = history or []
        # Optionally remove diacritics
        question = remove_diacritics(question)

        # Pass the question and conversation history to your chain
        response = get_answer(self.chain, question, history)
        history.append((question, response))
        return history, "", history

with gr.Blocks(title="Ahkam Bot - أحكام شرعية بوت") as demo:
    # 1. HTML Header with the embedded image
    gr.HTML(
        f"""
        <div class="header-container" style="
            background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%);
            padding: 2rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div class="header-content" style="display: flex; align-items: center; gap: 2rem;">
                <img src="{university_logo}" alt="University Logo" 
                     style="width: 100px; height: auto; border-radius: 50%; border: 3px solid white;" />
                <div class="header-text" style="color: white;">
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
                        Ahkam Bot - أحكام شرعية بوت
                    </h1>
                    <p style="margin: 0.5rem 0 0; font-size: 1.1rem; opacity: 0.9;">
                        جامعة طرابلس - كلية الهندسة - EE569
                    </p>
                </div>
            </div>
            <div class="welcome-message" style="max-width: 1200px; margin: 1.5rem auto 0; text-align: center; color: white;">
                <p style="font-size: 1.8rem; margin: 0; font-weight: 700; color: #ffd700;">
                    بسم الله الرحمن الرحيم
                </p>
                <p style="font-size: 1.2rem; margin: 1rem 0 0; line-height: 1.6;">
                    أهلاً وسهلاً بك في بوت الأحكام الشرعية. يمكنك طرح أسئلتك حول العبادات اليومية.
                </p>
            </div>
        </div>
        """
    )

    chatbot = gr.Chatbot(label="المحادثة", show_label=False)

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                placeholder="اكتب سؤالك هنا... مثال: ما هي شروط صحة الصلاة؟",
                lines=2,
                max_lines=4,
                autofocus=True
            )
        with gr.Column(scale=1):
            openai_api_key = gr.Textbox(
                type="password",
                label="مفتاح OpenAI API",
                placeholder="أدخل مفتاح API الخاص بك هنا..."
            )
            submit_btn = gr.Button("إرسال السؤال ➡️")

    state = gr.State()

    question.submit(
        fn=Chat(config=default_config),
        inputs=[question, state, openai_api_key],
        outputs=[chatbot, question, state],
    )
    submit_btn.click(
        fn=Chat(config=default_config),
        inputs=[question, state, openai_api_key],
        outputs=[chatbot, question, state],
    )

    
    gr.HTML(
        """
        <div class="footer" style="text-align: center; margin-top: 2rem;">
            <p>مشرف المادة: د. نوري بن بركة | الطلبة: طلال بادي - الطاهر صالح - بدر شحيم</p>
        </div>
        """
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="localhost",
        server_port=8844,
        show_error=True
    )
