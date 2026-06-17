import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from src import config
from src.retriever import ThresholdRetriever

def load_system_prompt():
    """
    Loads the system prompt from prompts/system_prompt.txt.
    Falls back to a default prompt if the file is not found.
    """
    if os.path.exists(config.SYSTEM_PROMPT_PATH):
        with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    # Fallback default prompt
    return """You are a precise, helpful FAQ assistant.
Your sole purpose is to answer questions using the uploaded company FAQ document provided to you as context.
You do not use any external knowledge, personal opinions, or information beyond the retrieved FAQ chunks.

## Strict Grounding Rules
1. ONLY answer using the [CONTEXT] blocks provided.
2. If the answer is not present, respond with: "I'm sorry, I couldn't find an answer to that in our FAQ. Please contact support at support@company.com"
3. Never guess, infer, or extrapolate beyond what is explicitly stated in the context.
4. Never reveal internal chunk IDs, scores, or metadata to the user.
5. If multiple chunks provide partial answers, synthesize them into one coherent response.
6. Keep answers concise: 2–4 sentences unless the question requires steps.
7. Always cite which FAQ section the answer comes from.
8. Match the user's input language automatically.
9. Temperature must be set to 0.1 or lower to prevent hallucination."""

class ConversationalFAQChain:
    """
    Wrapper class to intercept queries and use the LLM to condense
    conversational history + follow-up questions into standalone search queries.
    """
    def __init__(self, retrieval_qa_chain, condense_llm):
        self.retrieval_qa = retrieval_qa_chain
        self.condense_llm = condense_llm

    def invoke(self, inputs):
        query = inputs["query"]
        chat_history = inputs.get("chat_history", [])
        
        # If there is no chat history, search with the original query
        if not chat_history:
            return self.retrieval_qa.invoke({"query": query})
            
        # Format chat history for context
        formatted_history = ""
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted_history += f"{role}: {msg.get('content', '')}\n"
            
        condense_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be searched in an FAQ database. If the follow-up question is already a standalone question, output it unchanged. Do not answer the question, just output the rephrased question.

Chat History:
{formatted_history}
Follow-Up Question: {query}
Standalone Question:"""
        
        try:
            rephrased_response = self.condense_llm.invoke(condense_prompt)
            rephrased_query = rephrased_response.content.strip()
            if not rephrased_query:
                rephrased_query = query
        except Exception:
            rephrased_query = query
            
        # Run retrieval QA with the standalone query
        return self.retrieval_qa.invoke({"query": rephrased_query})


def build_rag_chain(vector_store, api_key: str = None):
    """
    Builds the complete RAG chain using Groq's ChatGroq LLM,
    ThresholdRetriever, and the ConversationalFAQChain query rephraser.
    """
    # Use provided API key or fall back to environment variable
    groq_api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your environment, "
            ".env file, or enter it in the interface."
        )
    
    # Initialize the LLM
    llm = ChatGroq(
        model_name=config.GROQ_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=groq_api_key
    )
    
    # Initialize our custom ThresholdRetriever
    retriever = ThresholdRetriever(vector_store=vector_store)
    
    # Construct the consolidated prompt template dynamically.
    # If the user has already included the {context} variable in the system prompt file,
    # we only need to append the question block. Otherwise, we append both.
    system_prompt = load_system_prompt()
    if "{context}" in system_prompt:
        full_prompt_template = system_prompt + "\n\nQuestion: {question}\n\nAnswer:"
    else:
        full_prompt_template = (
            system_prompt + 
            "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    
    prompt = PromptTemplate(
        template=full_prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Create the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return ConversationalFAQChain(qa_chain, llm)

