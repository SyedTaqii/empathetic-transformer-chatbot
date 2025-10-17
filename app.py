import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import string
from tokenizers import BertWordPieceTokenizer

# *** FIX: Moved st.set_page_config() to be the first Streamlit command ***
st.set_page_config(layout="wide")

# --- Caching the model and tokenizer for performance ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load the Transformer model and tokenizer once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer_path = "./tokenizer/vocab.txt"
    tokenizer = BertWordPieceTokenizer(tokenizer_path, lowercase=False)
    vocab_size = tokenizer.get_vocab_size()

    # Define model parameters
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_FF, DROPOUT = 512, 2, 2, 2, 2048, 0.1
    
    # Instantiate the model using your notebook's class structure
    model = Transformer(vocab_size, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_FF, DROPOUT)
    
    # Load the saved state dictionary
    try:
        # Using 'model.pt' as requested
        model_path = 'model.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file not found. Please make sure '{model_path}' is in the same directory.")
        return None, None, None, None
        
    model.to(device)
    model.eval()
    
    special_tokens = {
        'pad': tokenizer.token_to_id('[PAD]'),
        'bos': tokenizer.token_to_id('[BOS]'),
        'eos': tokenizer.token_to_id('[EOS]'),
        'all': ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[BOS]', '[EOS]']
    }

    return model, tokenizer, special_tokens, device

# =============================================================================
# SECTION 1: YOUR NOTEBOOK'S CLASSES AND HELPER FUNCTIONS
# =============================================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(f"([{string.punctuation}])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_mask):
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt, encoder_output, tgt_mask, src_mask):
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))
        cross_attn_output = self.cross_attn(tgt, encoder_output, encoder_output, src_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # *** FIX: Using 'encoder_layers' and 'decoder_layers' to match the saved model file ***
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    # --- Using bug-fixed encode/decode methods for correct inference ---
    def encode(self, src, src_mask):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        return encoder_output

    def decode(self, tgt, encoder_output, tgt_mask, src_mask):
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        return self.fc_out(decoder_output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.fc_out.weight.device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def create_padding_mask(self, seq, pad_token_id):
        return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)


def greedy_decode(model, src, max_len, start_symbol, device, pad_token_id, eos_token_id):
    src = src.to(device)
    src_mask = model.create_padding_mask(src, pad_token_id)
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, encoder_output, tgt_mask, src_mask)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == eos_token_id: break
    return ys

def beam_search_decode(model, src, max_len, start_symbol, device, beam_size, pad_token_id, eos_token_id):
    src = src.to(device)
    src_mask = model.create_padding_mask(src, pad_token_id)
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
    beams = [(torch.tensor([start_symbol], device=device).unsqueeze(0), 0.0)]
    for _ in range(max_len - 1):
        new_beams = []
        all_ended = True
        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                new_beams.append((seq, score)); continue
            all_ended = False
            with torch.no_grad():
                tgt_mask = model.generate_square_subsequent_mask(seq.size(1)).to(device)
                out = model.decode(seq, encoder_output, tgt_mask, src_mask)
                logits = out[:, -1]
                log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=1)
            for i in range(beam_size):
                next_word_idx = top_indices[0, i].item()
                new_beams.append((torch.cat([seq, torch.tensor([[next_word_idx]], device=device)], dim=1), score + top_log_probs[0, i].item()))
        if all_ended: break
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    return beams[0][0]

# =============================================================================
# SECTION 2: STREAMLIT UI
# =============================================================================

# --- Load resources ---
model, tokenizer, special_tokens, device = load_model_and_tokenizer()

# --- App Title and Description ---
st.title("🤖 Empathetic Conversational Chatbot")
st.markdown("This chatbot is built from scratch using a Transformer model. It's designed to generate empathetic replies based on a given situation and emotion.")

# --- Sidebar Controls ---
st.sidebar.header("Chat Controls")
emotion_list = [
    'sentimental', 'afraid', 'proud', 'faithful', 'terrified', 'joyful',
    'angry', 'sad', 'jealous', 'grateful', 'prepared', 'embarrassed',
    'excited', 'annoyed', 'lonely', 'ashamed', 'guilty', 'surprised',
    'nostalgic', 'confident', 'furious', 'disappointed', 'caring',
    'trusting', 'disgusted', 'anticipating', 'anxious', 'hopeful',
    'impressed', 'content'
]
selected_emotion = st.sidebar.selectbox("Choose an Emotion", sorted(list(set(emotion_list))))
decoding_strategy = st.sidebar.radio("Decoding Strategy", ('Beam Search', 'Greedy Search'))

# --- Main Chat Interface ---
if model is not None:
    # Initialize session state for conversation history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Get situation from user
    situation = st.text_area("Enter a situation to start the conversation:", "I was at the park with my best friend and we saw a dog.")

    # Display conversation history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What did you say?"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            strategy = 'beam' if decoding_strategy == 'Beam Search' else 'greedy'
            
            normalized_situation = normalize_text(situation)
            normalized_utterance = normalize_text(prompt)
            input_text = f"Emotion: {selected_emotion} | Situation: {normalized_situation} | Customer: {normalized_utterance} Agent:"
            
            encoded_input = tokenizer.encode(input_text)
            src = torch.tensor([special_tokens['bos']] + encoded_input.ids + [special_tokens['eos']], dtype=torch.long).unsqueeze(0).to(device)

            if strategy == 'beam':
                output_ids = beam_search_decode(model, src, max_len=50, start_symbol=special_tokens['bos'], device=device, beam_size=3, pad_token_id=special_tokens['pad'], eos_token_id=special_tokens['eos'])
            else: # greedy
                output_ids = greedy_decode(model, src, max_len=50, start_symbol=special_tokens['bos'], device=device, pad_token_id=special_tokens['pad'], eos_token_id=special_tokens['eos'])

            response_text = tokenizer.decode(output_ids.squeeze(0).tolist())
            for token in special_tokens['all']:
                response_text = response_text.replace(token, '')
            response_text = response_text.strip()

        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.history.append({"role": "assistant", "content": response_text})
else:
    st.warning("Model is not loaded. Please check the file paths and try again.")