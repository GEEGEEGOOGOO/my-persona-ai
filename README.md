# The Adaptive Loyalist - An AI Persona Engine

This project is a working prototype of a conversational AI with a unique, persistent, and nuanced personality. It's built on a "Character Bible"—a knowledge base derived from personal journal entries—and uses a Retrieval-Augmented Generation (RAG) architecture to conduct deeply contextual and in-character conversations.

This is the V1 prototype of "The Adaptive Loyalist," a persona that is self-aware, regulated, and deeply loyal, but has a defined breaking point at betrayal.

---

## ## Key Features

* **Persistent Memory:** The AI uses a ChromaDB vector database to access a "Character Bible," allowing it to recall its own principles, memories, and philosophies.
* **Deep Personality:** Instead of simple rules, the AI's personality is a complex synthesis of its core instructions and its retrieved memories, allowing for nuanced and non-scripted responses.
* **Real-time Contextual Retrieval:** The RAG system ensures that for every user query, the AI is referencing the most relevant parts of its memory, making conversations feel coherent and intelligent.
* **Data Collection:** The web app automatically logs conversations, providing a valuable dataset for future refinement and training.

---

## ## Technology Stack

This project was built using the following key technologies:

* **Language:** Python
* **LLM:** Google's Gemini API (gemini-1.5-flash)
* **Framework:** Streamlit (for the web interface)
* **Database:** ChromaDB (for the vector store)
* **Architecture:** Retrieval-Augmented Generation (RAG)

---

## ## How to Run Locally

1.  Clone this repository.
2.  Install the required libraries: `pip install -r requirements.txt`
3.  Set up your Gemini API key in `.streamlit/secrets.toml`.
4.  Run the application: `streamlit run webapp.py`
