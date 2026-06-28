LangChain Translator

A simple Python application that uses LangChain and Groq's Llama 3.3 model to translate text into various languages.

Features:

Utilizes ChatGroq for high-speed inference.
Implements ChatPromptTemplate for structured prompt management.

Uses StrOutputParser for clean, readable output.

Implements LangChain Expression Language (LCEL) and advanced **Runnables** for modular workflows:

*   **RunnablePassthrough**: Dynamically injects context and data into the chain on the fly.
*   **RunnableLambda**: Integrates custom Python functions directly into the execution pipeline.
*   **RunnableParallel**: Executes multiple distinct translation or analysis paths concurrently.
*   **RunnableWithMessageHistory**: Tracks and maintains stateful conversational memory across sessions.