# NLP Ahkam Sharia Chatbot with Retrieval-Augmented Generation (RAG)

**Deep Learning – EE569**

## Overview

This repository contains the implementation of an LLM-powered Ahkam Chatbot designed to answer questions related to دات ا (Ibadat) as part of the EE569 assignment. The chatbot leverages a Retrieval-Augmented Generation (RAG) approach by first retrieving relevant information from a vector database and then using a large language model (LLM) to generate responses. This project encompasses the following key tasks:

1. **Collect Relevant Data:**  
   Scrape data from the Aldorar website using BeautifulSoup, process it, and save it in JSON format. Diacritical marks were removed to improve retrieval performance.

2. **Create an Evaluation Dataset:**  
   Build a dataset of questions and answers to evaluate chatbot performance. The dataset includes questions generated from both the collected data and alternative sources.

3. **Create Proper Prompts:**  
   Design system and evaluation prompts to guide the chatbot’s responses and its evaluation process, ensuring the bot responds appropriately to sensitive topics.

4. **Embed Data into a Vector Database:**  
   Convert the processed data into embeddings using OpenAI’s text-embedding-ada-002-v2 model and store them in a vector database. Data batching and removal of diacritical marks were applied to enhance token efficiency and retrieval accuracy.

5. **Integrate LangChain:**  
   Build a retrieval-based pipeline using LangChain’s RetrievalQA module to fetch relevant information from the vector store and combine it with the LLM’s generative capabilities.

6. **Build a Web Interface:**  
   Develop a user-friendly interface with Gradio that allows users to interact with the chatbot and view a history of queries and responses.

7. **Evaluate the Chatbot:**  
   Assess the chatbot’s performance using both automated evaluation (via LangChain’s QAEvalChain) and human assessment. Key evaluation metrics include accuracy, response quality, and error handling.

For more details on each task, refer to the assignment document.

## Repository Structure

```
.
├── data
│   ├── collected_data.json       # Data collected from web scraping
│   ├── evaluation_dataset.json   # Dataset for chatbot evaluation
├── LLM
│   ├── app.py                    # Main application integrating LangChain and Gradio
│   ├── evaluation.py             # Script for evaluating the chatbot
│   └── chatbot.py                # Chatbot implementation using LangChain’s retrieval chain
├── prompts
│   ├── system_prompt.json        # System prompt guiding chatbot behavior
│   └── evaluation_prompt.json    # Evaluation prompt for grading responses
├── embeddings
│   └── vector_store.pkl          # Serialized vector database with embeddings
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/talalbadi/NLP-Akmam-Sharia-chatbot-RAG-.git
   cd NLP-Akmam-Sharia-chatbot-RAG-
   ```

2. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys:**
   If your project requires API keys (e.g., for OpenAI), set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

### Running the Chatbot

To launch the chatbot with the Gradio web interface, run:
```bash
python LLM/app.py
```
Then, open the provided URL in your browser to start interacting with the chatbot.

### Evaluation

To evaluate the chatbot using the prepared evaluation dataset, execute:
```bash
python LLM/evaluation.py
```
This script will generate responses for the evaluation questions and compute performance metrics, which are logged in an evaluation report.

## Project Details

- **Data Collection:**  
  Data was scraped from the [Aldorar website](https://dorar.net/) and processed by removing diacritical marks to improve retrieval performance.

- **Embedding & Retrieval:**  
  OpenAI’s text-embedding-ada-002-v2 model was used to generate embeddings. These embeddings were then stored in a vector database, with careful batching and preprocessing for optimal performance.

- **LangChain Integration:**  
  The project integrates LangChain to build a retrieval pipeline that fetches relevant context from the vector database and feeds it into the LLM to generate accurate answers.

- **Web Interface:**  
  A simple Gradio-based web interface enables users to input queries, receive responses, and view interaction history in real time.

- **Evaluation:**  
  The chatbot was evaluated using both automated methods (via LangChain’s QAEvalChain) and human assessment. Special consideration was given to correct handling of uncertainty (e.g., when the chatbot responds with “I don’t know the answer”).

## Contributors

- Talal Malek Badi – 2200208609
- Badr Shehim – 2200208256
- Ataher Saleh – 2200208085

**Supervisor:** Dr. Nuri Benbarka  
**Assignment Date:** February 16, 2025

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the OpenAI API team for providing the language models.
- Thanks to the LangChain developers for the robust retrieval-based framework.
- Special thanks to all contributors and reviewers who supported this project.
