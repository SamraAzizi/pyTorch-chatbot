#  Neural Chatbot Assistant

A PyTorch-based chatbot that classifies user intents and responds appropriately, with support for custom function execution.

##  Prerequisites

- Python 3.8+
- PyTorch
- NLTK

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-assistant.git
   cd chatbot-assistant
   ```

2. Install dependencies:
```bash
pip install torch nltk numpy
```

3. Download NLTK data:
```bash
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Model Architecture
```bash
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        self.fc1 = nn.Linear(input_size, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)          # Hidden layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
```

## Project Structure
```bash
chatbot-assistant/
├── intents.json            # Intent patterns and responses
├── main.py                # Main implementation
└── README.md              
```
