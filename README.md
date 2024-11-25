# OnDemand Professor Q&A Bot

This project implements an OnDemand Q&A bot to assist professors in retrieving information from lecture slides and documents. It leverages natural language processing and vector-based document retrieval to respond to questions accurately.

## Requirements

### Environment
- **Operating System**: Windows/Linux/MacOS
- **Python Version**: 3.8 or higher
- **Virtual Environment**: (optional) Recommended to use `venv` or `conda` for managing dependencies.

### Setup
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd OnDemand-Professor-Q-A-Bot
2. Install required libraries
3. Environment Variables:
  Set up the OPEN AI API token by adding it to the environment variables:
    ```bash
    `export OPENAI_API_KEY_TOKEN='your-openai-api-key'`

### Adopted Libraries
The following libraries are used in this project:

**Chainlit**: Provides real-time message handling and interaction.

**LangChain**: Powers LLM-based document retrieval and Q&A functionality.

**Chroma**: Used for vector storage of document embeddings.

**Hugging Face Hub**: For loading transformer models and embeddings.

**PyPDFLoader**: For loading PDF documents from a directory.

### System Architecture
![image](https://github.com/user-attachments/assets/8311c280-2827-415b-98bd-cfc8d39742f9)

## Flow of Execution

### Document Loading
- The `DirectoryLoader` and `PyPDFLoader` modules are used to read and process lecture slides and documents stored in the `Lectures` folder.
- All documents are parsed and prepared for embedding and retrieval.

### Text Splitting and Embedding
- Documents are split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
- Each chunk is embedded into a vector representation using OpenAI's `OpenAIEmbeddings` for efficient similarity searches.

### Vector Store and Retrieval
- The `Chroma` library is used as a vector store to save the embedded chunks, enabling fast and accurate retrieval based on the query.
- When a query is received, the most relevant document chunks are retrieved from the vector store.

### Question Answering (QA)
- The `ChatOpenAI` model (GPT-3.5 Turbo) is used to generate accurate and context-aware answers.
- Retrieved document chunks are passed to the model to provide a precise response with relevant context.

---


## Commands to Run the Code
### Start the Q&A Bot
To start the bot, use the following command:
    
    chainlit run app.py -w

## Issues and Solutions

### Current Issues

#### Redundant or Repetitive Responses
- **Issue**: Responses sometimes include redundant or repetitive information.
- **Solution**: The response formatting has been improved to remove redundant context.

#### Incorrect Source Information for Certain Queries
- **Issue**: The bot provides irrelevant source information for queries that do not rely on the database.
- **Solution**: Logic has been added to suppress source information when it is not applicable.

#### Starting Page Index
- **Issue**: The source documents' page numbers start from 0 instead of 1.
- **Solution**: The page indexing has been updated to start from 1 to match user expectations.

## Suggestions and Feedback

- **Use of Larger Models**: Upgrading to GPT-4 can improve the bot's accuracy and provide more nuanced answers, especially for complex queries, if budget permits.
- **Improved Document Support**: Adding support for additional document formats such as DOCX and HTML will enhance the bot’s ability to handle a wider range of content.
- **Detailed Error Logging**: Incorporating more detailed logging mechanisms will improve debugging and provide better insights into any issues encountered.

---

## Future Enhancements

1. **Support for Real-Time Document Updates**: Implement functionality to allow real-time additions or updates to the vector store, eliminating the need to restart the bot for new documents.
2. **Integration with Other Platforms**: Extend the bot’s usability by integrating it with platforms such as Slack, Microsoft Teams, or other communication tools.
3. **Advanced Retrieval Techniques**: Combine keyword-based and vector-based retrieval methods to improve the relevance and accuracy of retrieved document chunks.
