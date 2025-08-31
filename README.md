# AI Search Assistant

A research chat application that combines real-time web search with local AI analysis using Streamlit, LangChain, and Ollama.

## Features

- Real-time web search integration
- Local AI-powered response generation
- Optimized for speed and performance
- Clean chat interface
- No external API keys required
- Privacy-focused (all processing local)

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for better performance)
- Internet connection for web search
- Windows, macOS, or Linux

### Required Software
- **Ollama**: Local LLM inference engine
- **Git**: For repository management
- **Python**: Programming language runtime

## Installation

### Step 1: Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai and install
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
# Download from https://ollama.ai and install
# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Install AI Model

```bash
# Start Ollama service
ollama serve

# In a new terminal, pull the model
ollama pull llama3.1

# Verify installation
ollama list
```

### Step 3: Clone Repository

```bash
git clone https://github.com/dripston/aisearchassistant.git
cd aisearchassistant
```

### Step 4: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the Application

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the app
streamlit run app.py
```

### Access the Interface

Open your web browser and navigate to:
```
http://localhost:8501
```

### Basic Usage

1. Type your question in the chat input
2. Wait for "Typing..." indicator
3. Receive AI-analyzed search results
4. Continue the conversation

## Code Explanation

### Architecture Overview

The application uses a graph-based architecture with LangGraph to manage the conversation flow:

```
User Input → Web Search → Result Processing → LLM Analysis → Response
```

### Core Components

#### 1. Search Engine (`improved_search` function)

Implements multiple search strategies with fallback mechanisms:

```python
def improved_search(query: str) -> str:
    # Method 1: Google News RSS feed
    # Method 2: DuckDuckGo Instant Answer API  
    # Method 3: Traditional DuckDuckGo search
    # Method 4: Fallback error handling
```

**Search Flow:**
- Attempts Google News RSS parsing for recent news
- Falls back to DuckDuckGo API for general queries
- Uses traditional web search as final fallback
- Implements 10-second timeout for reliability

#### 2. Result Processing (`extract_key_info` function)

Optimizes search results for LLM consumption:

```python
def extract_key_info(search_result: str) -> str:
    # Remove noise (URLs, short fragments)
    # Split into meaningful sentences
    # Take first 3-4 relevant sentences
    # Truncate to 800 characters maximum
```

**Processing Steps:**
1. Clean whitespace and formatting
2. Filter out URLs and irrelevant content
3. Extract sentences longer than 15 characters
4. Limit to 3-4 most relevant sentences
5. Truncate to prevent LLM overload

#### 3. LLM Configuration

Optimized Ollama settings for speed:

```python
llm = ChatOllama(
    model="llama3.1",        # Base model
    temperature=0.1,         # Low randomness for consistency
    num_predict=200,         # Maximum response tokens
    num_ctx=2048            # Context window size
)
```

**Configuration Explanation:**
- **model**: Uses Llama 3.1 (8B parameters) for good quality/speed balance
- **temperature**: Low value (0.1) for consistent, factual responses
- **num_predict**: Limits response to 200 tokens (~150 words) for speed
- **num_ctx**: Reduces context window to 2048 tokens for faster processing

#### 4. State Management

Uses LangGraph for conversation state:

```python
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: State):
    # Extract user question
    # Perform web search
    # Process results
    # Generate LLM response
    # Return updated state
```

**State Flow:**
1. Receives conversation history
2. Extracts latest user message
3. Performs web search with optimization
4. Creates minimal prompt for LLM
5. Returns updated conversation state

#### 5. Frontend Interface

Streamlit-based chat interface:

```python
# Message display loop
for msg in st.session_state.lc_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input handling
user_input = st.chat_input("Type your message...")
if user_input:
    # Add to conversation
    # Process through graph
    # Display response
```

## Configuration Options

### Model Selection

Change the AI model by modifying `app.py`:

```python
# Fast, lightweight option
llm = ChatOllama(model="phi3")

# Ultra-fast, minimal option  
llm = ChatOllama(model="gemma2:2b")

# High-quality option (slower)
llm = ChatOllama(model="llama3.1:70b")
```

### Performance Tuning

Adjust LLM parameters for your needs:

```python
llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,         # 0.0 = deterministic, 1.0 = creative
    num_predict=100,         # Shorter responses = faster
    num_ctx=1024,           # Smaller context = faster
    top_p=0.8,              # Nucleus sampling parameter
    repeat_penalty=1.1      # Prevent repetitive responses
)
```

### Search Optimization

Modify search result limits:

```python
def extract_key_info(search_result: str, max_chars: int = 800):
    # Increase max_chars for more detailed results
    # Decrease for faster processing
```

## Troubleshooting

### Common Issues

**1. Slow Response Times (>30 seconds)**

```bash
# Check Ollama status
ollama ps

# Restart Ollama
ollama restart

# Try faster model
ollama pull phi3
# Then update app.py model parameter
```

**2. Search Failures**

```python
# Check internet connection
# Verify DuckDuckGo access
# Review console logs for specific errors
```

**3. Memory Issues**

```bash
# Monitor system resources
# Close other applications
# Try smaller model:
ollama pull gemma2:2b
```

**4. Module Import Errors**

```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python version
python --version
```

### Performance Optimization

**For Faster Responses:**
1. Use smaller models (phi3, gemma2:2b)
2. Reduce `num_predict` to 50-100 tokens
3. Lower `num_ctx` to 1024 or 512
4. Set `temperature=0` for deterministic responses

**For Better Quality:**
1. Use larger models (llama3.1:70b)
2. Increase `num_predict` to 500+ tokens
3. Raise `temperature` to 0.3-0.7
4. Expand `num_ctx` to 4096+

### Alternative Models

```bash
# Very fast models
ollama pull phi3           # 3.8B parameters
ollama pull gemma2:2b      # 2B parameters
ollama pull qwen2:1.5b     # 1.5B parameters

# Balanced models  
ollama pull llama3.1:8b    # 8B parameters (default)
ollama pull mistral:7b     # 7B parameters

# High-quality models (slower)
ollama pull llama3.1:70b   # 70B parameters
ollama pull mixtral:8x7b   # 46.7B parameters
```

## Development

### Project Structure

```
aisearchassistant/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── .gitignore         # Git ignore rules
```

### Code Architecture

**Graph-Based Flow:**
```python
User Input → Search → Processing → LLM → Response
```

**Key Functions:**
- `improved_search()`: Multi-source web search with fallbacks
- `extract_key_info()`: Search result optimization
- `chatbot()`: Main conversation handler
- Streamlit UI: Frontend interface

### Extending Functionality

**Add New Search Sources:**
```python
def improved_search(query: str) -> str:
    # Add your custom search implementation
    try:
        # Custom API or scraping logic
        pass
    except:
        # Fallback to existing methods
        pass
```

**Custom Response Processing:**
```python
def custom_processor(search_result: str) -> str:
    # Add domain-specific processing
    # Filter for specific content types
    # Apply custom formatting
    return processed_result
```

## Deployment

### Local Deployment

```bash
# Standard local run
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8080

# External access
streamlit run app.py --server.address 0.0.0.0
```

### Production Considerations

1. **Resource Management**: Monitor CPU/RAM usage with multiple users
2. **Rate Limiting**: Implement search rate limits for production
3. **Error Handling**: Enhanced logging and error recovery
4. **Security**: Input validation and sanitization
5. **Monitoring**: Application performance metrics

## Contributing

### Development Setup

```bash
# Fork the repository
git clone https://github.com/yourusername/aisearchassistant.git
cd aisearchassistant

# Create feature branch
git checkout -b feature-name

# Make changes and test
streamlit run app.py

# Commit and push
git add .
git commit -m "Add: feature description"
git push origin feature-name

# Create pull request on GitHub
```

### Code Style Guidelines

- Use meaningful variable names
- Add type hints for function parameters
- Include docstrings for complex functions
- Follow PEP 8 style guidelines
- Add error handling for external API calls

### Testing

```bash
# Test different models
ollama pull phi3
# Update app.py model parameter and test

# Test search functionality
# Try various query types (news, facts, technical terms)

# Test error conditions
# Disconnect internet, stop Ollama, test recovery
```

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Ollama documentation: https://ollama.ai/docs
3. Check LangChain documentation: https://langchain.readthedocs.io
4. Open an issue on GitHub

## Changelog

### v1.0.0
- Initial release
- Basic search and chat functionality
- Ollama integration
- Streamlit interface

### Performance Benchmarks

**Typical Response Times:**
- Search: 1-3 seconds
- LLM Processing: 5-30 seconds (depending on model)
- Total: 6-33 seconds per query

**Model Comparison:**
- **phi3**: 5-10 seconds average
- **llama3.1:8b**: 15-30 seconds average  
- **gemma2:2b**: 3-8 seconds average
