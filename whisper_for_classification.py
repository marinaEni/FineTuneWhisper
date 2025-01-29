import torch.nn as nn
from transformers import WhisperModel

# https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md


class WhisperForClassification(nn.Module):
    def __init__(self, model_name: str, n_out: int):
        super(WhisperForClassification, self).__init__()
        self.n_out = n_out
        
        # Load the Whisper model
        print("Load the Whisper model")
        whisper_model = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_model.encoder
        
        print("Freeze the Whisper encoder parameters")
        # Freeze the Whisper encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 
                      self.encoder.config.hidden_size), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.encoder.config.hidden_size, 
                      self.encoder.config.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.encoder.config.hidden_size//2, n_out),
            )
        
        if n_out == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
    
    # ---------------------------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask=None): # Whether to generate an attention mask in the feature extractor
        # Get encoder outputs
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # Use the hidden states from the encoder
        hidden_states = encoder_outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)
        
        # Pool the outputs (use the mean of the hidden states) # Apply mean pooling over the sequence dimension to get (batch_size, hidden_size)
        pooled_output = hidden_states.mean(dim=1)
        
        # Pass through the classifier
        logits = self.classifier(pooled_output)
        
        # Apply the activation function
        output = self.activation(logits) # probabilities
        
        return output
    # ---------------------------------------------------------------------------------------------
    def predict(self, input_features, attention_mask=None, th=0.5):
        # Forward pass to get probabilities
        probs = self.forward(input_features, attention_mask)
        
        if self.n_out == 1:
            # Convert probabilities to binary class labels using a threshold
            preds = (probs > th).long()
        else:
            # For multi-class classification, get the class with the highest probability
            preds = probs.argmax(dim=1)
        
        return probs, preds
"""
Since all elements in the batch are padded/truncated to a maximum length in the input space, 
we don't require an attention mask when forwarding the audio inputs to the Whisper model. 
Whisper is unique in this regard - with most audio models, you can expect to provide an attention
mask that details where sequences have been padded, and thus where they should be ignored in the
self-attention mechanism. Whisper is trained to operate without an attention mask and infer directly
from the speech signals where to ignore the inputs.

Enhanced Model Understanding: Attention mechanisms in neural networks allow the model to focus
on specific parts of the input data when making predictions. By generating an attention mask,
you enable the model to provide insights into which parts of the input it finds most relevant 
for a given task. This can be valuable for understanding the model’s decision-making process 
and debugging.

Handling Variable-Length Inputs: Attention masks are often used in sequence-to-sequence models 
(e.g., machine translation, speech recognition) to handle variable-length inputs effectively. 
They allow the model to focus on different parts of the input sequence during encoding, 
which is particularly useful when dealing with sequences of varying lengths.
"""    
        
# # Example usage
# model_name = "openai/whisper-tiny"
# num_out = 1  # For binary classification, use 1; for multi-class classification, use the number of classes
# model = WhisperForClassification(model_name, num_out)
# print(model)
# # Sample input
# input_ids = torch.randn(1, 80, 3000)  # Example input tensor
# attention_mask = torch.ones_like(input_ids)  # Example attention mask

# # Forward pass
# model.eval()
# output = model(input_ids, attention_mask)
# print(output)
