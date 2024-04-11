# Installation

### Clone the repository:
`git clone https://github.com/BaoTranHuyDuc/RAG_customer_service_chatbot.git`

### Navigate to the project directory:
`cd your-repository`

### Install all the required packages using pip and the `requirements.txt` file:
`pip install -r requirements.txt`

### Download punkt for TruLens"
`python download_punkt.py`

### Set up OpenAI API key
`touch .env`

Paste your OpenAI key into this .env file:

`OPENAI_API_KEY="sk-..."`

# Create vector store
Use SQL to extract a csv of product descriptions from the company database.

Name this file `product_desc\product.csv`

Run this file through the notebook `product_desc_cleaning.ipynb`

Run `create-vectorstore.py`:
`python create-vecstore.py`

Note that this vector store current use the RecursiveTextSplitter to chunk text. I find that using Semantic Chunking (https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/) creates more semantically meaningful chunks but these chunks are larger and hence use more tokens.

# Run the app
In the terminal, run:
`python -m streamlit run app.py`

The streamlit app for that chat bot is at: http://localhost:8501/
The streamlit app for the TruLens dashboard is at: http://localhost:8502/

# Errors with the demo:

Currently, when you evaluate this demo with the TruLens dashboard, the groundedness measure is usually 0. I believe that this is because the LLM response and the context are in Vietnamese while the evaluating prompts are in English which makes it hard for the evaluating LLM to find information overlap between the response and the context. I believe that this can be fixed if we can change the TruLens prompts to Vietnamese.

# Experience from building this chatbot:
I have note down my experience with building RAG chatbot in this document for future interns who wish to continue this project: https://docs.google.com/document/d/1i8-gAn5m6Nd6OTDa0VDT1pogw5nW5UzD-lgLCyl1hu8/edit 
