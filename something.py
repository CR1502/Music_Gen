import torch
import torch.nn as nn
from musiclm_pytorch import MusicLM

# Load the tokenized data
data = torch.load("tokenized_data.pt")

# Define the parameters for the language model
input_size = len(data.vocab)
hidden_size = 512
num_layers = 2
dropout = 0.2

# Instantiate the MusicLM model
model = MusicLM(input_size, hidden_size, num_layers, dropout)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, batch in enumerate(data):
        # Get the input and target sequences
        input_seq = batch[:-1]
        target_seq = batch[1:]
        
        # Forward pass
        output = model(input_seq)
        loss = criterion(output.reshape(-1, input_size), target_seq.reshape(-1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data)}], Loss: {loss.item():.4f}")
            
# Save the trained model
torch.save(model.state_dict(), "musiclm_model.pt")