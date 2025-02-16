# -*- coding: utf-8 -*-

import os
from types import SimpleNamespace
import gradio as gr
import wandb
import re
from chain import get_answer, load_chain, load_vector_store
from config import default_config

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

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "llmapps"

class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""
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
        if openai_api_key:
            openai_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("الرجاء إدخال مفتاح OpenAI API الخاص بك")

        if self.vector_store is None:
            self.vector_store = load_vector_store(self.wandb_run, openai_key)
        if self.chain is None:
            self.chain = load_chain(self.wandb_run, self.vector_store, openai_key)

        history = history or []
        question =remove_diacritics(question)
        response = get_answer(self.chain, question, history)
        history.append((question, response))
        return history, "", history

with gr.Blocks(theme=gr.themes.Soft(), title="Ahkam Bot - أحكام شرعية بوت") as demo:
    gr.HTML(
        """
        <div class="header-container">
            <div class="header-content">
                <img src="LLM/university_logo.png" alt="University Logo" class="logo"/>
                <div class="header-text">
                    <h1 class="title">Ahkam Bot - أحكام شرعية بوت</h1>
                    <p class="subtitle">جامعة طرابلس - كلية الهندسة - EE569</p>
                </div>
            </div>
            <div class="welcome-message">
                <p class="bismillah">بسم الله الرحمن الرحيم</p>
                <p class="instructions">أهلاً وسهلاً بك في بوت الأحكام الشرعية. يمكنك طرح أسئلتك حول العبادات اليومية.</p>
            </div>
        </div>
        """
    )

    gr.HTML(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');

            :root {
                --primary-color: #2d6a4f;
                --secondary-color: #40916c;
                --background-color: #f8f9fa;
                --text-color: #2d3436;
                --arabic-font: 'Noto Sans Arabic', sans-serif;
            }

            body {
                font-family: var(--arabic-font), Arial, sans-serif;
            }

            .header-container {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                padding: 2rem;
                border-radius: 0 0 20px 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }

            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                gap: 2rem;
            }

            .logo {
                width: 100px;
                height: auto;
                border-radius: 50%;
                border: 3px solid white;
            }

            .header-text {
                color: white;
            }

            .title {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                font-family: var(--arabic-font);
            }

            .subtitle {
                margin: 0.5rem 0 0;
                font-size: 1.1rem;
                opacity: 0.9;
                font-family: var(--arabic-font);
            }

            .welcome-message {
                max-width: 1200px;
                margin: 1.5rem auto 0;
                text-align: center;
                color: white;
                font-family: var(--arabic-font);
            }

            .bismillah {
                font-size: 1.8rem;
                margin: 0;
                font-weight: 700;
                color: #ffd700;
            }

            .instructions {
                font-size: 1.2rem;
                margin: 1rem 0 0;
                line-height: 1.6;
            }

            .rtl-textbox {
                text-align: right;
                direction: rtl;
            }

            .rtl-chatbot .bot, .rtl-chatbot .user {
                direction: rtl;
                text-align: right;
            }
        </style>
        """
    )

    with gr.Row(variant="panel"):
        chatbot = gr.Chatbot(
            elem_classes="chatbot rtl-chatbot",
            label="المحادثة",
            show_label=False,
            bubble_full_width=False,
        )

    with gr.Row(variant="panel"):
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="",
                placeholder="اكتب سؤالك هنا... مثال: ما هي شروط صحة الصلاة؟",
                elem_id="input-box",
                lines=2,
                max_lines=4,
                autofocus=True,
                elem_classes=["rtl-textbox"]
            )
        with gr.Column(scale=1):
            openai_api_key = gr.Textbox(
                type="password",
                label="مفتاح OpenAI API",
                placeholder="أدخل مفتاح API الخاص بك هنا...",
                elem_id="key-box"
            )
            submit_btn = gr.Button("إرسال السؤال ➡️", elem_classes="submit-btn")

    state = gr.State()

    question.submit(
        Chat(config=default_config),
        [question, state, openai_api_key],
        [chatbot, question, state],
    )
    submit_btn.click(
        Chat(config=default_config),
        [question, state, openai_api_key],
        [chatbot, question, state],
    )

    gr.HTML(
        """
        <div class="footer">

            <p>مشرف المادة: د. نوري بن بركة |  الطلبة: طلال بادي - الطاهر صالح - بدر شحيم</p>
        </div>
        """
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="localhost",
        server_port=8844,
        show_error=True,
     
    )
