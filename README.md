Empathetic Transformer Chatbot

This project is a complete implementation of a Transformer-based conversational chatbot built from scratch in PyTorch. The model is trained on the Empathetic Dialogues dataset to generate empathetic replies based on a user's situation and emotional context. The final application is deployed as an interactive web app using Streamlit.

This repository was created as a submission for the "Project 02: Empathetic Conversational Chatbot" assignment.

Live Demo

The chatbot is deployed on Streamlit Cloud and is publicly accessible:

>> Launch the Chatbot Demo <<

(Note: Please replace the URL above with your actual Streamlit deployment link.)

Features

Empathetic Responses: The model generates replies tailored to the emotional context provided by the user.

Built from Scratch: The Transformer encoder-decoder architecture is implemented entirely from scratch using PyTorch, with no reliance on pre-trained models.

Interactive UI: A user-friendly web interface built with Streamlit allows for real-time conversation.

Conversation History: The application maintains and displays the history of the current conversation.

Selectable Decoding Strategy: Users can switch between Greedy Search and Beam Search to compare the quality of different text generation methods.

Model Architecture

The chatbot is powered by a standard Transformer encoder-decoder model. All weights were randomly initialized and trained end-to-end.

The key components, all implemented from scratch, include:

WordPiece Tokenization trained on the source corpus.

Sinusoidal Positional Encodings.

Multi-Head Attention mechanisms (including a causal mask in the decoder).

Position-wise Feed-Forward Networks.

Residual Connections and Layer Normalization.

The final trained model (model.pt) uses the following hyperparameters:

Embedding Dimension (d_model): 512

Number of Heads (nhead): 2

Encoder Layers: 2

Decoder Layers: 2

Dropout Rate: 0.1

Dataset

The model was trained on the Empathetic Dialogues dataset provided by Facebook AI. The dataset contains over 25,000 conversations grounded in emotional situations.

Source: Kaggle - Empathetic Dialogues

Data Split: The data was split into 80% for training, 10% for validation, and 10% for testing.

Setup and Installation

To run this project on a local machine, please follow these steps.

Prerequisites

Python 3.8+

Git

Git LFS (for downloading the large model file)

Installation

Clone the repository:

git clone [https://github.com/your-username/empathetic-transformer-chatbot.git](https://github.com/your-username/empathetic-transformer-chatbot.git)
cd empathetic-transformer-chatbot


Install Git LFS:
Make sure Git LFS is installed. You can download it from git-lfs.github.com. After installing, run:

git lfs install


Pull LFS files:
Download the model.pt file from Git LFS.

git lfs pull


Install Python dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt


How to Run

Once the setup is complete, you can launch the Streamlit application with the following command:

streamlit run app.py


The application should open automatically in your web browser at http://localhost:8501.

Other Deliverables

Evaluation Report: Link to Evaluation Report (Placeholder)

Blog Post Draft: Link to Medium Blog Post (Placeholder)