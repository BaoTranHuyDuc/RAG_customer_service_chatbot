## Installation

### Clone the repository:
`git clone https://github.com/yourusername/your-repository.git`

### Navigate to the project directory:
`cd your-repository`

### Install all the required packages using pip and the `requirements.txt` file:
`pip install -r requirements.txt`

## Create vector store
Use SQL to extract a csv of product descriptions from the company database.

Name this file `product_desc\product.csv`

Run this file through the notebook `product_desc_cleaning.ipynb`

Run `create-vectorstore.py`:
`python create-vecstore.py`

## Run the app
In the terminal, run:
`python -m streamlit run app.py`

The streamlit app for that chat bot is at: `http://localhost:8501/`

The streamlit app for the TruLens dashboard is at: `http://localhost:8502/`
