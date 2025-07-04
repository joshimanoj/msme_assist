import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from data_loader import load_rag_data, load_dfl_data
from utils import (
    get_embeddings,
    extract_scheme_guid,
)
import streamlit as st
from data import DataManager
import re
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize DataManager
data_manager = DataManager()

# Initialize cached resources
@st.cache_resource
def init_llm():
    logger.info("Initializing LLM client")
    start_time = time.time()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")
    llm = ChatOpenAI(
        model="grok-3-mini-fast",
        reasoning_effort="low",
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        temperature=0
    )
    logger.info(f"LLM initialized in {time.time() - start_time:.2f} seconds")
    return llm

@st.cache_resource
def init_vector_store():
    logger.info("Loading vector store")
    start_time = time.time()
    vector_store = load_rag_data(faiss_index_path="faiss_index", version_file="faiss_version.txt")
    logger.info(f"Vector store loaded in {time.time() - start_time:.2f} seconds with {vector_store.index.ntotal} documents")
    return vector_store

@st.cache_resource
def init_dfl_vector_store():
    logger.info("Loading DFL vector store")
    start_time = time.time()
    google_drive_file_id = os.getenv("DFL_GOOGLE_DOC_ID")
    if not google_drive_file_id:
        raise ValueError("DFL_GOOGLE_DOC_ID environment variable not set")
    vector_store = load_dfl_data(google_drive_file_id)
    logger.info(
        f"DFL vector store loaded in {time.time() - start_time:.2f} seconds with {vector_store.index.ntotal} documents"
    )
    return vector_store

llm = init_llm()
scheme_vector_store = init_vector_store()
dfl_vector_store = init_dfl_vector_store()
embeddings = get_embeddings()

# Dataclass to hold user context information
@dataclass
class UserContext:
    name: str
    state_id: str
    state_name: str
    business_name: str
    business_category: str
    gender: str

# Retrieve user information from Streamlit session state
def get_user_context(session_state):
    try:
        user = session_state.user
        return UserContext(
            name=user["fname"],
            state_id=user.get("state_id", "Unknown"),
            state_name=user.get("state_name", "Unknown"),
            business_name=user.get("business_name", "Unknown"),
            business_category=user.get("business_category", "Unknown"),
            gender=user.get("gender", "Unknown"),
        )
    except AttributeError:
        logger.error("User data not found in session state")
        return None

# Helper function to detect language
def detect_language(query):
    # Check for Devanagari script (Hindi)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(query):
        return "Hindi"
    
    # Common Hindi words in Roman script for Hinglish detection
    hindi_words = [
        "kya", "kaise", "ke", "mein", "hai", "kaun", "kahan", "kab",
        "batao", "sarkari", "yojana", "paise", "karobar", "dukaan", "nayi", "naye", "chahiye", "madad", "karo",
        "dikhao", "samjhao", "tarika", "aur", "arey", "bhi", "kya", "hai", "hoga", "hogi", "ho", "hoon", "magar", "lekin", "par", 
        "toh", "ab", "phir", "kuch", "thoda", "zyada", "sab", "koi", "kuchh", "aap", "tum", "main",
        "hum", "unhe", "unko", "unse", "yeh", "woh", "aisa", "aisi", "aise", "bataiye", "achha", "acha", "accha", "theek", "theekh", 
        "thik", "thikk", "idhar", "udhar", "yahan", "wahan", "waha", "bhai", "bhaiya", "bhaiyya", 
        "bhiya", "bahut", "bahot", "bohot", "bahuut", "zara", "jara", "mat", "maat", "matlab", "matlb", "fir", "phirr", "phhir", "phir", 
        "main", "aap", "aapke", "yojanaen", "liye", "kar", "sakte", "hain", "tak"
    ]
    query_lower = query.lower()
    hindi_word_count = sum(1 for word in hindi_words if word in query_lower)
    total_words = len(query_lower.split())
    
    # If more than 30% of words are Hindi or mixed with English
    if total_words > 0 and hindi_word_count / total_words > 0.15:
        return "Hinglish"

    return "English"

def get_system_prompt(language, user_name="User"):

    """Return tone and style instructions."""

    system_rules = f"""1. **Language Handling**:
       - The query language is provided as {language} (English, Hindi, or Hinglish).
       - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
       - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script, prioritizing hindi words in the mix.
       - For English queries, respond in simple English.
       
       2. **Response Guidelines**:
       - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
       - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
       - Response must be at least 200 words and ≤500 words.
       - Never mention agent fees unless specified in RAG Response for scheme queries.
       - Use scheme names exactly as provided in the RAG Response without paraphrasing (underscores may be replaced with spaces).
       - Start the response with 'Hi {user_name}!' (English), 'Namaste {user_name}!' (Hinglish), or 'नमस्ते {user_name}!' (Hindi) unless Out_of_Scope."""

    system_prompt = system_rules.format(language=language, user_name=user_name)
    return system_prompt


# Build conversation history from stored messages for intent classification
def build_conversation_history(messages):
    conversation_history = ""
    session_messages = []
    for msg in messages[-10:]:
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
    session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
    for role, content, _ in session_messages:
        conversation_history += f"{role.capitalize()}: {content}\n"
    return conversation_history


# Welcome user
def welcome_user(state_name, user_name, query_language):
    """Generate a welcome message in the user's chosen language."""
    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth. The user is a new user named {user_name} from {state_name}.

    **Input**:
    - Query Language: {query_language}

    **Instructions**:
    - Generate a welcome message for a new user in the specified language ({query_language}).
    - For Hindi, use Devanagari script with simple, clear words suitable for micro business owners with low Hindi proficiency.
    - For English, use simple English with a friendly tone.
    - The message should welcome the user, mention their state ({state_name}), and offer assistance with schemes and documents applicable to their state and all central government schemes or help with digital/financial literacy and business growth.
    - Response must be ≤70 words.
    - Start the response with 'Hi {user_name}!' (English) or 'नमस्ते {user_name}!' (Hindi).

    **Output**:
    - Return only the welcome message in the specified language.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated_response = response.content.strip()
        logger.info(f"Generated welcome message in {query_language}: {generated_response}")
        return generated_response
    except Exception as e:
        logger.error(f"Failed to generate welcome message: {str(e)}")
        # Fallback to default messages
        if query_language == "Hindi":
            return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। आप {state_name} से हैं, मैं आपकी राज्य और केंद्रीय योजनाओं में मदद करूँगा। आज कैसे सहायता करूँ?"
        return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! Since you're from {state_name}, I'll help with schemes and documents applicable to your state and all central government schemes. How can I assist you today?"

# Step 1: Process user query with RAG
def get_rag_response(query, vector_store, state="ALL_STATES", gender=None, business_category=None):
    start_time = time.time()
    try:
        details = []
        if state:
            details.append(f"state: {state}")
        if gender:
            details.append(f"gender: {gender}")
        if business_category:
            details.append(f"business category: {business_category}")

        fallback_details = []
        if state:
            fallback_details.append(f"state: {state}")
        if gender:
            fallback_details.append(f"gender: {gender}")

        fallback_query = query
        full_query = query
        if details:
            full_query = f"{full_query}. {' '.join(details)}"
        if fallback_details:
            fallback_query = f"{fallback_query}. {' '.join(fallback_details)}"

        logger.debug(f"Processing query: {full_query}")
        embed_start = time.time()
        query_embedding = embeddings.embed_query(full_query)
        logger.debug(f"Query embedding generated in {time.time() - embed_start:.2f} seconds (first 10 values): {query_embedding[:10]}")
        retrieve_start = time.time()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": full_query})
        logger.debug(f"Retrieval and QA completed in {time.time() - retrieve_start:.2f} seconds")
        response = result["result"]
        sources = result["source_documents"]
        logger.info(f"RAG response generated in {time.time() - start_time:.2f} seconds: {response}")
        if not sources:
            logger.warning(
                f"No documents retrieved for query with profile details: {full_query}."
            )
            if full_query != query:
                logger.info("Retrying RAG search without profile details")
                result = qa_chain.invoke({"query": query})
                response = result["result"]
                sources = result["source_documents"]
        if not sources:
            logger.warning(f"No documents retrieved for query: {query}")
            logger.info(f"Attempting fallback retrieval with query: {fallback_query}")
            result = qa_chain.invoke({"query": fallback_query})
            response = result["result"]
            sources = result["source_documents"]
        if not sources:
            logger.warning(f"Fallback also returned no documents for query: {query}")
            return "No relevant scheme information found."
        logger.info(f"Retrieved {len(sources)} documents for query: {query}")
        for i, doc in enumerate(sources):
            logger.debug(f"Document {i+1}:")
            logger.debug(f"  Content: {doc.page_content}")
            logger.debug(f"  Metadata: {doc.metadata}")
        return {"text": response, "sources": sources}
    except Exception as e:
        logger.error(f"RAG retrieval failed in {time.time() - start_time:.2f} seconds: {str(e)}")
        return {"text": "Error retrieving scheme information.", "sources": []}


def get_scheme_response(
    query,
    vector_store,
    state="ALL_STATES",
    gender=None,
    business_category=None,
    include_mudra=False,
):
    """Wrapper for scheme dataset retrieval with optional Mudra scheme info."""
    logger.info("Querying scheme dataset")
    rag = get_rag_response(
        query,
        vector_store,
        state=state,
        gender=gender,
        business_category=business_category,
    )

    if include_mudra:
        logger.info("Fetching Pradhan Mantri Mudra Yojana details")
        mudra_rag = get_rag_response(
            "Pradhan Mantri Mudra Yojana",
            vector_store,
            state=state,
            gender=gender,
            business_category=business_category,
        )

        if not isinstance(rag, dict):
            rag = {"text": str(rag), "sources": []}
        if not isinstance(mudra_rag, dict):
            mudra_rag = {"text": str(mudra_rag), "sources": []}

        rag["text"] = f"{rag.get('text', '')}\n{mudra_rag.get('text', '')}"
        rag["sources"] = rag.get("sources", []) + mudra_rag.get("sources", [])

    return rag


def get_dfl_response(query, vector_store, state="ALL_STATES", gender=None, business_category=None):
    """Wrapper for DFL dataset retrieval with clearer logging."""
    logger.info("Querying DFL dataset")
    return get_rag_response(
        query,
        vector_store,
        state=state,
        gender=gender,
        business_category=business_category,
    )

# Check query similarity for context
def is_query_related(query, prev_query, prev_response):
    prompt = f"""You are an assistant for Haqdarshak, helping small business owners in India with government schemes, digital/financial literacy, and business growth. Determine if the current query is a follow-up to the previous conversation.

    **Input**:
    - Current Query: {query}
    - Previous User Query: {prev_query}
    - Previous Bot Response: {prev_response}

    **Instructions**:
    - A query is a related follow-up if it is ambiguous and contextually refers to one of the schemes or topics mentioned in the previous interaction.
    - Examples of ambiguous queries: 'Tell me more', 'How to apply?', 'What next?', 'Can you help with it?', 'और बताएं', 'आगे क्या?'.
    - The query is a follow-up if it seeks clarification or additional information about a previously discussed scheme or topic.
    - The query is not a follow-up if it introduces a new scheme or topic not mentioned above (e.g., 'What is FSSAI?', 'How to use UPI?', 'एफएसएसएआई क्या है?') or is unrelated (e.g., 'What’s the weather?', 'मौसम कैसा है?', 'Time?').
    - Return 'True' if the query is a follow-up, 'False' otherwise. Focus on the previous interaction only, ignoring rule-based keyword matching or similarity scores.

    **Output**:
    - Return only 'True' or 'False'.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        result = response.content.strip()
        logger.debug(
            f"LLM determined query '{query}' is {'related' if result == 'True' else 'not related'} to previous context"
        )
        return result == "True"
    except Exception as e:
        logger.error(f"Failed to determine query relation: {str(e)}")
        return False

# Classify the intent of the user's query
def classify_intent(query, prev_response, conversation_history=""):
    """Return one of the predefined intent labels."""

    prompt = f"""You are an assistant for Haqdarshak. Classify the user's intent.

    **Input**:
    - Query: {query}
    - Previous Assistant Response: {prev_response}
    - Conversation History: {conversation_history}

    **Instructions**:
    Return only one label from the following:
       - Specific_Scheme_Know_Intent (e.g., 'What is FSSAI?', 'PMFME ke baare mein batao', 'एफएसएसएआई क्या है?', 'Pashu Kisan Credit Scheme ke baare mein bataiye', 'Udyam', 'Mudra Yojana')
       - Specific_Scheme_Apply_Intent (e.g., 'Apply', 'Apply kaise karna hai', 'How to apply for FSSAI?', 'FSSAI kaise apply karu?', 'एफएसएसएआई के लिए आवेदन कैसे करें?')
       - Specific_Scheme_Eligibility_Intent (e.g., 'Eligibility', 'Eligibility batao', 'Am I eligible for FSSAI?', 'FSSAI eligibility?', 'एफएसएसएआई की पात्रता क्या है?')
       - Schemes_Know_Intent (e.g., 'Schemes for credit?', 'MSME ke liye schemes kya hain?', 'क्रेडिट के लिए योजनाएं?', 'loan chahiye', 'scheme dikhao')
       - DFL_Intent (digital/financial literacy queries, e.g., 'How to use UPI?', 'UPI kaise use karein?', 'डिजिटल भुगतान कैसे करें?', 'Opening Bank Account', 'Why get Insurance', 'Why take loans', 'Online Safety', 'How can going digital help grow business', etc.)
       - Out_of_Scope (e.g., 'What's the weather?', 'Namaste', 'मौसम कैसा है?', 'Time?')
       - Contextual_Follow_Up (e.g., 'Tell me more', 'Aur batao', 'और बताएं', 'iske baare mein aur jaankaari chahiye')
       - Confirmation_New_RAG (Only to be chosen when user query is confirmation for initating another RAG search ("Yes", "Haan batao", "Haan dikhao", "Yes search again") AND previous assistant response says that the bot needs to fetch more details about some scheme. ('I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.')
       - Gratitude_Intent (user expresses thanks or acknowledgement, e.g., 'ok thanks', 'got it', 'theek hai', 'accha', 'thank you', 'शुक्रिया', 'धन्यवाद')

    **Tips**:
       - Use rule-based checks for Out_of_Scope (keywords: 'hello', 'hi', 'hey', 'weather', 'time', 'namaste', 'mausam', 'samay').
       - For Contextual_Follow_Up, prioritize the Previous Assistant Response for context to check if the query is a follow-up.
       - Only use conversation history for context, intent should be determined solely by current query.
       - To distinguish between Specific_Scheme_Know_Intent and Scheme_Know_Intent, check for whether query is asking for information about specific scheme or general information about schemes. You can also refer to conversation history to see if the scheme being asked about has already been mentioned by the bot to the user first, in which case the intent is certainly Specific_Scheme_Know_Intent.
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

# Generate final response based on intent and RAG output
def generate_response(intent, rag_response, user_info, language, context, scheme_guid=None, scheme_details=None):
    if intent == "Out_of_Scope":
        if language == "Hindi":
            return "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।"
        if language == "Hinglish":
            return "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon."
        return "Sorry, I can only help with government schemes, digital/financial literacy or business growth."

    if intent == "Gratitude_Intent":
        gratitude_prompt = f"""You are a friendly assistant for Haqdarshak. The user {user_info.name} has thanked you.

        **Instructions**:
        - Respond briefly in the same language ({language}) acknowledging the thanks and offering further help.
        - Use Devanagari script for Hindi and a natural mix of Hindi and English words in Roman script for Hinglish.
        - Keep the message under 30 words.

        **Output**:
        - Only the acknowledgement message in the user's language."""
        try:
            response = llm.invoke([{"role": "user", "content": gratitude_prompt}])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate gratitude response: {str(e)}")
            if language == "Hindi":
                return "धन्यवाद! क्या मैं और मदद कर सकता हूँ?"
            if language == "Hinglish":
                return "Thanks! Kya main aur madad kar sakta hoon?"
            return "You're welcome! Let me know if you need anything else."

    tone_prompt = get_system_prompt(language, user_info.name)

    base_prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth.

    **Input**:
    - Intent: {intent}
    - RAG Response: {rag_response}
    - User Name: {user_info.name}
    - State: {user_info.state_name} ({user_info.state_id})
    - Gender: {user_info.gender}
    - Business Name: {user_info.business_name}
    - Business Category: {user_info.business_category}
    - Conversation Context: {context}
    - Language: {language}"""
    if scheme_details:
        base_prompt += f"\n    - User Provided Scheme Details: {scheme_details}"
    if scheme_guid:
        base_prompt += f"\n    - Scheme GUID: {scheme_guid}"

    base_prompt += f"""

    **Language Handling and Tone Instructions**:
    {tone_prompt}

    **Task**:
    Use any user-provided scheme details to pick relevant schemes from retrieved data and personalise the scheme information wherever applicable.
    """

    special_schemes = ["Udyam", "FSSAI", "Shop Act", "GST"]
    link = "https://haqdarshak.com/contact"

    if intent == "Specific_Scheme_Know_Intent":
        intent_prompt = (
            "Share scheme name, purpose, benefits from **RAG Response** (≤120 words). "
            "Filter for schemes where 'applicability' includes state_id or 'ALL_STATES' "
            "or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). List CSS schemes first, "
            "then state-specific. Ask: 'Want details on eligibility or how to apply?' "
            "(English), 'Eligibility ya apply karne ke baare mein jaanna chahte hain?' "
            "(Hinglish), or 'पात्रता या आवेदन करने के बारे में जानना चाहते हैं?' (Hindi)."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you get this document "
            f"for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein "
            f"dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi)."
        )
    elif intent == "Specific_Scheme_Apply_Intent":
        intent_prompt = (
            "Share application process from **RAG Response** (≤120 words). Filter for schemes "
            "where 'applicability' includes state_id or 'ALL_STATES' or 'scheme type' is "
            "'Centrally Sponsored Scheme' (CSS)."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you get this document "
            f"for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein "
            f"dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi)."
        )
    elif intent == "Specific_Scheme_Eligibility_Intent":
        intent_prompt = (
            "Summarize eligibility rules from **RAG Response** (≤120 words) and provide a link "
            f"to check eligibility: https://customer.haqdarshak.com/check-eligibility/{scheme_guid}. "
            "Ask the user to verify their eligibility there."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you get this document "
            f"for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein "
            f"dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi)."
        )
    elif intent == "Schemes_Know_Intent":
        intent_prompt = (
            "List schemes from **RAG Response** (2-3 lines each, ≤120 words). Filter for schemes "
            "where 'applicability' includes state_id or 'ALL_STATES' or 'scheme type' is "
            "'Centrally Sponsored Scheme' (CSS). Use any user provided scheme details to choose the most relevant schemes. "
            "If no close match is found, still list the top 2-3 schemes applicable to the user that are at least in the user's state or CSS. Finally Ask: 'Want more details on any scheme?' "
            "(English), 'Kisi yojana ke baare mein aur jaanna chahte hain?' (Hinglish), or "
            "'किसी योजना के बारे में और जानना चाहते हैं?' (Hindi)."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: 'Haqdarshak can help you get this document "
            f"for Only ₹99. Click: {link}' (English), 'Haqdarshak aapko yeh document sirf ₹99 mein "
            f"dilane mein madad kar sakta hai. Click: {link}' (Hinglish), or 'हकदर्शक आपको यह दस्तावेज़ "
            f"केवल ₹99 में दिलाने में मदद कर सकता है। क्लिक करें: {link}' (Hindi)."
        )
        if scheme_details and scheme_details.get("path") == "credit":
            intent_prompt += (
                " Always include 'Pradhan Mantri Mudra Yojana' in the same format as the other schemes."
            )
    elif intent == "DFL_Intent":
        intent_prompt = (
            "Use the **RAG Response** if available, augmenting with your own knowledge "
            "where relevant. If the RAG Response is empty, do not mention its absence; "
            "answer smoothly from your own knowledge in simple language (≤120 words) "
            "with helpful examples."
        )
    elif intent == "Contextual_Follow_Up":
        intent_prompt = (
            "Use the Previous Assistant Response and Conversation Context to identify the topic. "
            "If the RAG Response does not match the referenced scheme, indicate a new RAG search "
            "is needed. Provide a relevant follow-up response (≤120 words) using the RAG Response, "
            "filtering for schemes where 'applicability' includes state_id or 'scheme type' is "
            "'Centrally Sponsored Scheme' (CSS). If unclear, ask for clarification (e.g., "
            "'Could you specify which scheme?' or 'Kaunsi scheme ke baare mein?' or 'कौन सी योजना के बारे में?')."
        )
    elif intent == "Confirmation_New_RAG":
        intent_prompt = (
            "If the user confirms to initiate a new RAG search, respond with the details of the "
            "scheme they are interested in, refer to conversation context for details."
        )
    else:
        intent_prompt = ""

    output_prompt = """
    **Output**:
       - Return only the final response in the query's language (no intent label or intermediate steps). If a new RAG search is needed for schemes, indicate with: 'I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.' (English), 'Mujhe [scheme name] ke baare mein aur jaankari leni hogi. Kya aap isi scheme ki baat kar rahe hain?' (Hinglish), or 'मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?' (Hindi).
       - If RAG Response is empty or 'No relevant scheme information found,' and the query is a Contextual_Follow_Up referring to a specific scheme, indicate a new RAG search is needed. Otherwise, say: 'I don't have information on this right now.' (English), 'Mujhe iske baare mein abhi jaankari nahi hai.' (Hinglish), or 'मुझे इसके बारे में अभी जानकारी नहीं है।' (Hindi).
       - Do not mention any other scheme when a specific scheme is being talked about.
       - When intent is Schemes_Know, do not mention other schemes from past conversation, only the current relevant ones.
       - No need to mention user profile details in every response, only include where contextually relevant.
       - Scheme answers must come only from scheme data. For DFL answers, use the DFL document supplemented by your own knowledge when possible, but rely on your own knowledge if nothing relevant is found.
    """

    prompt = f"{base_prompt}{intent_prompt}\n{output_prompt}"

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        final_text = response.content.strip()
        if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
            screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
            if screening_link not in final_text:
                final_text += f"\n{screening_link}"
        return final_text
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        if language == "Hindi":
            return "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।"
        if language == "Hinglish":
            return "Sorry, main aapka query process nahi kar saka."
        return "Sorry, I couldn't process your query."

# Generate unique interaction ID
def generate_interaction_id(query, timestamp):
    return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"


# ---- Scheme personalisation helpers ----
def ask_scheme_question(key, language):
    questions_en = {
        "credit_or_subsidy": "Are you looking for loans/credit or other subsidies?",
        "loan_amount": "How much loan are you looking for?",
        "loan_purpose": "What is the purpose of taking this loan?",
        "turnover": "How much business do you do in a month approximately?",
        "business_type": "What business do you do?",
        "sc_st": "Are you SC/ST?",
    }

    questions_hi = {
        "credit_or_subsidy": "क्या आप लोन/क्रेडिट लेना चाहते हैं या कोई सब्सिडी?",
        "loan_amount": "आप कितने रुपये का लोन चाहते हैं?",
        "loan_purpose": "यह लोन किस काम के लिए है?",
        "turnover": "आप महीने में लगभग कितना व्यापार करते हैं?",
        "business_type": "आप कौन सा व्यापार करते हैं?",
        "sc_st": "क्या आप SC/ST हैं?",
    }

    questions_hinglish = {
        "credit_or_subsidy": "Kya aap loan/credit chahte hain ya koi subsidy?",
        "loan_amount": "Kitna loan chahiye?",
        "loan_purpose": "Yeh loan kis kaam ke liye hai?",
        "turnover": "Aap mahine mein lagbhag kitna business karte hain?",
        "business_type": "Aap kaunsa business karte hain?",
        "sc_st": "Kya aap SC/ST hain?",
    }

    if language == "Hindi":
        return questions_hi.get(key, "")
    if language == "Hinglish":
        return questions_hinglish.get(key, "")
    return questions_en.get(key, "")


def monthly_to_annual_llm(amount_str: str) -> str:
    """Use the LLM to convert a monthly amount description to an annual value."""
    prompt = (
        "You will be given a description of how much business a user does in a month. "
        "Convert it to an approximate annual amount in Indian rupees. "
        "Return only the number without commas, units, or any additional text.\n\n"
        f"Text: {amount_str}"
    )
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        normalized = response.content.strip().replace(",", "")
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", normalized)
        if match:
            return match.group(1)
    except Exception as e:
        logger.error(f"LLM amount normalisation failed: {e}")
    return amount_str


def classify_scheme_type(query: str) -> str:
    """Return 'credit' if query refers to loans/credit, else 'non_credit'."""
    credit_keywords = [
        "loan",
        "credit",
        "udhaar",
        "udhar",
        "borrow",
        "capital",
        "finance",
        "fund",
        "money",
        "purchase",
        "buy",
        "\u0932\u094b\u0928",  # लोन (loan)
        "\u0915\u0930\u094d\u091c",  # कर्ज (karz)
        "\u0909\u0927\u093e\u0930",  # उधार (udhaar)
        "\u090b\u0923",  # ऋण (riN)
    ]
    non_credit_keywords = [
        "document",
        "registration",
        "legal",
        "mentorship",
        "training",
        "license",
        "certificate",
        "\u0926\u0938\u094d\u0924\u093e\u0935\u0947\u091c",  # document in Hindi
        "\u0930\u091c\u093f\u0938\u094d\u091f\u094d\u0930\u0947\u0936\u0928",  # registration
        "\u0915\u093e\u0928\u0942\u0928\u0940",  # legal
        "\u092e\u0947\u0902\u091f\u094b\u0930\u0936\u093f\u092a",  # mentorship
        "\u092a\u094d\u0930\u0936\u093f\u0915\u094d\u0937\u0923",  # training
        "\u0932\u093e\u0907\u0938\u0947\u0902\u0938",  # license
        "\u092a\u094d\u0930\u092e\u093e\u0923\u092a\u0924\u094d\u0930",  # certificate
    ]
    q_lower = query.lower()
    if any(kw in q_lower for kw in credit_keywords):
        return "credit"
    if any(kw in q_lower for kw in non_credit_keywords):
        return "non_credit"
    return "credit"


def extract_scheme_names(text: str) -> list:
    """Use the LLM to extract scheme names from a text block.

    Returns a list of names or an empty list when no schemes are mentioned."""
    prompt = (
        "Extract the scheme names mentioned in the following text. "
        "Return the names exactly as written, separated by semicolons if there are multiple. "
        "If no scheme is present, reply with 'none'.\n\n"
        f"{text}"
    )
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        if not content or "none" in content.lower():
            logger.info("No scheme names extracted")
            return []
        names = [n.strip().replace("_", " ") for n in content.split(";") if n.strip()]
        logger.info(f"Extracted scheme names: {names}")
        return names
    except Exception as e:
        logger.error(f"Failed to extract scheme names: {e}")
        return []


def resolve_scheme_reference(query: str, scheme_names: list) -> str | None:
    """Determine which scheme from scheme_names the query refers to."""
    if not scheme_names:
        return None
    list_str = "; ".join(scheme_names)
    prompt = (
        "You will be given a user query and a list of scheme names. "
        "If the query refers to any of these schemes, either by name or even by order (first scheme, second scheme, pehli vaali, doosri vaali etc.), "
        "return the matching scheme name exactly as provided. If none match, return an empty string.\n\n"
        f"Query: {query}\nScheme_Names: {list_str}"
    )
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        name = response.content.strip().replace("_", " ")
        resolved = name or None
        logger.info(f"Resolved scheme reference '{query}' -> {resolved}")
        return resolved
    except Exception as e:
        logger.error(f"Failed to resolve scheme reference: {e}")
        return None


def handle_scheme_flow(answer, scheme_vector_store, session_id, mobile_number, user_info):
    language = st.session_state.scheme_flow_data.get("language", detect_language(answer))
    step = st.session_state.scheme_flow_step
    details = st.session_state.scheme_flow_data
    path = details.get("path")

    # Determine next step based on current state

    if details.get("path") == "credit":
        if step == 1:
            details["loan_amount"] = answer
            st.session_state.scheme_flow_step = 2
            return ask_scheme_question("loan_purpose", language), False
        if step == 2:
            details["loan_purpose"] = answer
            st.session_state.scheme_flow_step = 3
            return ask_scheme_question("turnover", language), False
        if step == 3:
            details["turnover"] = monthly_to_annual_llm(answer)
            st.session_state.scheme_flow_step = 4
            return ask_scheme_question("business_type", language), False
        if step == 4:
            details["business_type"] = answer
            st.session_state.scheme_flow_step = 5
            return ask_scheme_question("sc_st", language), False
        if step == 5:
            details["sc_st"] = answer
            st.session_state.scheme_flow_active = False
            st.session_state.scheme_flow_step = None
    else:
        if step == 1:
            details["turnover"] = monthly_to_annual_llm(answer)
            st.session_state.scheme_flow_step = 2
            return ask_scheme_question("business_type", language), False
        if step == 2:
            details["business_type"] = answer
            st.session_state.scheme_flow_step = 3
            return ask_scheme_question("sc_st", language), False
        if step == 3:
            details["sc_st"] = answer
            st.session_state.scheme_flow_active = False
            st.session_state.scheme_flow_step = None

    if st.session_state.scheme_flow_active:
        return "", False

    # Flow completed, fetch schemes using initial query
    query = details.get("initial_query", "")
    rag = get_scheme_response(
        query,
        scheme_vector_store,
        state=user_info.state_id,
        gender=user_info.gender,
        business_category=user_info.business_category,
        include_mudra=details.get("path") == "credit",
    )
    if isinstance(rag, dict):
        rag_text = rag.get("text")
    else:
        rag_text = rag
    response = generate_response(
        "Schemes_Know_Intent",
        rag_text or "",
        user_info,
        language,
        "",
        scheme_details=details,
    )
    names = extract_scheme_names(response)
    if names:
        st.session_state.scheme_names = names
        st.session_state.scheme_names_str = " ".join([f"{i}. {n}" for i, n in enumerate(names, 1)])
        logger.info(f"Stored scheme names after flow: {st.session_state.scheme_names_str}")
    return response, True

def generate_hindi_audio_script(original_response: str, user_info: UserContext) -> str:
    """
    Generates a summarized, human-like Hindi script for text-to-speech from the original bot response.
    The script should avoid punctuation marks and focus on natural flow.
    """
    prompt = f"""You are an assistant for Haqdarshak. Your task is to summarize the provided text into a concise, human-like script
    in natural Hindi (Devanagari script) for a text-to-speech system.
    
    **Instructions**:
    - Summarize the core information from the provided 'Original Response'.
    - Ensure the summary flows naturally as if spoken by a human.
    - Translate the summary into clear and simple Hindi (Devanagari script) using simple hindi words.
    - Focus on the main points and keep the summary concise, ideally under 150 words, to ensure a smooth audio experience.
    - The response should be purely the Hindi script, with no introductory or concluding remarks.
    - Do NOT use any english words. 
    - Do NOT translate Smileys or emoticons.
    - Always use simpler alternatives wherever the words are in complex hindi e.g. Instead of "vyavyasay" say "business", instead of "vanijya" say "finance"
    - Do NOT include urls and web links. 

    **Original Response**:
    {original_response}

    **Output**:
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        hindi_script = response.content.strip()
        logger.info(f"Generated Hindi audio script: {hindi_script}")
        return hindi_script
    except Exception as e:
        logger.error(f"Failed to generate Hindi audio script: {str(e)}")
        # Fallback: Attempt to translate the original response to Hindi as a last resort
        try:
            translation_prompt = f"Translate the following text into simple Hindi (Devanagari script), removing all punctuation and hyphens for a smooth audio output: {original_response}"
            translation_response = llm.invoke([{"role": "user", "content": translation_prompt}])
            hindi_script = translation_response.content.strip()
            logger.warning(f"Falling back to direct translation for Hindi audio script: {hindi_script}")
            return hindi_script
        except Exception as inner_e:
            logger.error(f"Failed to fall back to direct translation: {str(inner_e)}")
            return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।" # Error generating audio script.

# Main function to process query
def process_query(query, scheme_vector_store, dfl_vector_store, session_id, mobile_number, user_language=None):
    start_time = time.time()
    timings = {}

    def record(step_name, start):
        timings[step_name] = time.time() - start

    def log_timings():
        total = time.time() - start_time
        summary = "\n".join(f"{k}: {v:.2f}s" for k, v in timings.items())
        summary += f"\nTotal: {total:.2f}s"
        logger.info("Query processing timings:\n" + summary)

    logger.info(f"Starting query processing for: {query}")

    # Retrieve user data from session state using helper
    step = time.time()
    user_info = get_user_context(st.session_state)
    record("user_context", step)
    if not user_info:
        log_timings()
        return "Error: User not logged in.", None  # Return tuple
    user_name = user_info.name
    state_id = user_info.state_id
    state_name = user_info.state_name
    business_name = user_info.business_name
    business_category = user_info.business_category
    gender = user_info.gender

    if "scheme_names" not in st.session_state:
        st.session_state.scheme_names = []
    if "scheme_names_str" not in st.session_state:
        st.session_state.scheme_names_str = ""

    # Use user_language for welcome message, otherwise detect query language
    step = time.time()
    query_language = user_language if query.lower() == "welcome" and user_language else detect_language(query)
    record("language_detection", step)
    logger.info(f"Using query language: {query_language}")

    # Check user type and fetch recent conversations once
    step = time.time()
    conversations = data_manager.get_conversations(mobile_number, limit=8)
    user_type = "returning" if conversations else "new"
    record("fetch_conversations", step)
    logger.info(f"User type: {user_type}")

    if st.session_state.get("scheme_flow_active"):
        step = time.time()
        resp, _ = handle_scheme_flow(
            query,
            scheme_vector_store,
            session_id,
            mobile_number,
            user_info,
        )
        record("scheme_flow", step)
        generated_response = resp
        hindi_audio_script = generate_hindi_audio_script(generated_response, user_info)
        try:
            interaction_id = generate_interaction_id(query, datetime.utcnow())
            messages_to_save = [
                {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
                {"role": "assistant", "content": generated_response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script},
            ]
            if not any(
                any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
                for conv in conversations
            ):
                data_manager.save_conversation(session_id, mobile_number, messages_to_save)
            else:
                logger.debug(
                    f"Skipped saving duplicate conversation for query: {query} (Interaction ID: {interaction_id})"
                )
        except Exception as e:
            logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")
        log_timings()
        return generated_response, hindi_audio_script


    # Handle welcome query
    if query.lower() == "welcome":
        if user_type == "new":
            response = welcome_user(state_name, user_name, query_language)
            hindi_audio_script = generate_hindi_audio_script(response, user_info)
            try:
                interaction_id = generate_interaction_id(response, datetime.utcnow())
                if not any(
                    msg["role"] == "assistant" and msg["content"] == response
                    for conv in conversations for msg in conv["messages"]
                ):
                    data_manager.save_conversation(
                        session_id,
                        mobile_number,
                        [
                            {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script}
                        ]
                    )
                    logger.info(f"Saved welcome message for new user in session {session_id} (Interaction ID: {interaction_id})")
                else:
                    logger.debug(f"Skipped saving duplicate welcome message: {response}")
            except Exception as e:
                logger.error(f"Failed to save welcome message for new user in session {session_id}: {str(e)}")
            logger.info(f"Generated welcome response for new user in {time.time() - start_time:.2f} seconds: {response}")
            log_timings()
            return response, hindi_audio_script
        else:
            logger.info(f"No welcome message for returning user")
            log_timings()
            return None, None

    # Check if vector stores are valid
    step = time.time()
    try:
        doc_count = scheme_vector_store.index.ntotal
        logger.info(f"Scheme vector store contains {doc_count} documents")
        if doc_count == 0:
            logger.error("Vector store is empty")
            error_message = "No scheme data available. Please check the data source."
            if query_language == "Hindi":
                error_message = "कोई योजना डेटा उपलब्ध नहीं है। कृपया डेटा स्रोत की जाँच करें।"
            hindi_audio_script = generate_hindi_audio_script(error_message, user_info)
            record("vector_store_check", step)
            log_timings()
            return error_message, hindi_audio_script
        dfl_count = dfl_vector_store.index.ntotal
        logger.info(f"DFL vector store contains {dfl_count} documents")
        if dfl_count == 0:
            logger.error(
                "DFL vector store is empty, will rely solely on LLM knowledge"
            )
        record("vector_store_check", step)
    except Exception as e:
        logger.error(f"Vector store check failed: {str(e)}")
        error_message = "Error accessing scheme data."
        if query_language == "Hindi":
            error_message = "योजना डेटा तक पहुँचने में त्रुटि।"
        hindi_audio_script = generate_hindi_audio_script(error_message, user_info)
        record("vector_store_check", step)
        log_timings()
        return error_message, hindi_audio_script

    # Check if query is related to any previous query in the session
    scheme_rag = None
    dfl_rag = None
    related_prev_query = None
    session_cache = st.session_state.rag_cache.get(session_id, {})
    dfl_session_cache = st.session_state.dfl_rag_cache.get(session_id, {})

    # Get the most recent query-response pair from the current session
    recent_query = None
    recent_response = None
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                msg_index = st.session_state.messages.index(msg)
                if msg_index > 0 and st.session_state.messages[msg_index - 1]["role"] == "user":
                    recent_query = st.session_state.messages[msg_index - 1]["content"]
                break



    # Determine if the current query is a follow-up to the recent response
    follow_up = False
    if recent_query and recent_response:
        step = time.time()
        follow_up = is_query_related(
            query,
            recent_query,
            recent_response,
        )
        record("follow_up_check", step)

    # Use conversation context only when the query is a follow-up
    context_pair = f"User: {recent_query}\nAssistant: {recent_response}" if follow_up else ""

    step = time.time()
    conversation_history = build_conversation_history(st.session_state.messages)
    intent = classify_intent(query, recent_response or "", conversation_history)
    record("intent_classification", step)
    augmented_query = query
    logger.info(f"Using conversation context: {context_pair}")
    logger.info(f"Classified intent: {intent}")

    maintain_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
    if intent not in maintain_intents:
        st.session_state.scheme_flow_data = {}
        st.session_state.scheme_flow_active = False
        st.session_state.scheme_flow_step = None

    query_scheme_names = extract_scheme_names(query)
    if query_scheme_names:
        logger.info(f"Scheme names detected in query: {query_scheme_names}")
    stored_names = st.session_state.scheme_names
    referenced_scheme = None
    if intent in {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}:
        if query_scheme_names:
            match = None
            lower_stored = [n.lower() for n in stored_names]
            for nm in query_scheme_names:
                if nm.lower() in lower_stored:
                    match = nm
                    break
            if match:
                referenced_scheme = match
            else:
                referenced_scheme = query_scheme_names[0]
                stored_names = [referenced_scheme]
        else:
            referenced_scheme = resolve_scheme_reference(query, stored_names)
            if not referenced_scheme and stored_names:
                referenced_scheme = stored_names[0]
        if referenced_scheme:
            augmented_query = f"Referenced Scheme: {referenced_scheme}. {query}"
            st.session_state.scheme_names = stored_names if referenced_scheme in stored_names else [referenced_scheme]
            st.session_state.scheme_names_str = " ".join([f"{i}. {n}" for i, n in enumerate(st.session_state.scheme_names, 1)])
            logger.info(f"Updated stored scheme names: {st.session_state.scheme_names_str}")

    if intent == "Schemes_Know_Intent" and not st.session_state.scheme_flow_active:
        st.session_state.scheme_flow_active = True
        scheme_type = classify_scheme_type(query)
        st.session_state.scheme_flow_step = 1
        st.session_state.scheme_flow_data = {
            "initial_query": query,
            "language": query_language,
            "path": scheme_type,
        }
        if scheme_type == "credit":
            first_q = ask_scheme_question("loan_amount", query_language)
        else:
            first_q = ask_scheme_question("turnover", query_language)
        step = time.time()
        hindi_audio_script = generate_hindi_audio_script(first_q, user_info)
        record("scheme_flow", step)
        try:
            interaction_id = generate_interaction_id(query, datetime.utcnow())
            messages_to_save = [
                {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
                {"role": "assistant", "content": first_q, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script},
            ]
            if not any(
                any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
                for conv in conversations
            ):
                data_manager.save_conversation(session_id, mobile_number, messages_to_save)
        except Exception as e:
            logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")
        log_timings()
        return first_q, hindi_audio_script

    scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

    if intent in scheme_intents:
        if intent == "Specific_Scheme_Know_Intent":
            step = time.time()
            scheme_rag = get_scheme_response(
                query,
                scheme_vector_store,
                state=None,
                gender=None,
                business_category=None,
                include_mudra=False,
            )
            record("rag_retrieval", step)
        else:
            scheme_rag = session_cache.get(query)
    elif intent in dfl_intents:
        dfl_rag = dfl_session_cache.get(query)

    if follow_up and recent_query and recent_response:
        if intent in scheme_intents:
            scheme_rag = session_cache.get(recent_query, scheme_rag)
        elif intent in dfl_intents:
            dfl_rag = dfl_session_cache.get(recent_query, dfl_rag)
        related_prev_query = recent_query

    if scheme_rag is None and intent in scheme_intents:
        step = time.time()
        scheme_rag = get_scheme_response(
            augmented_query,
            scheme_vector_store,
            state=state_id,
            gender=gender,
            business_category=business_category,
            include_mudra=classify_scheme_type(augmented_query) == "credit",
        )
        record("rag_retrieval", step)
        if session_id not in st.session_state.rag_cache:
            st.session_state.rag_cache[session_id] = {}
        st.session_state.rag_cache[session_id][query] = scheme_rag

    if dfl_rag is None and intent in dfl_intents:
        step = time.time()
        dfl_rag = get_dfl_response(
            query,
            dfl_vector_store,
            state=state_id,
            gender=gender,
            business_category=business_category,
        )
        record("rag_retrieval", step)
        if session_id not in st.session_state.dfl_rag_cache:
            st.session_state.dfl_rag_cache[session_id] = {}
        st.session_state.dfl_rag_cache[session_id][query] = dfl_rag

    rag_response = scheme_rag if intent in scheme_intents else dfl_rag
    rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
    if intent == "DFL_Intent" and (
        rag_text is None or "No relevant" in rag_text
    ):
        rag_text = ""
    scheme_guid = None
    if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
        scheme_guid = extract_scheme_guid(rag_response.get("sources", []))
    step = time.time()
    generated_response = generate_response(
        intent,
        rag_text or "",
        user_info,
        query_language,
        context_pair,
        scheme_guid=scheme_guid,
        scheme_details=st.session_state.scheme_flow_data if st.session_state.get("scheme_flow_data") else None,
    )
    record("generate_response", step)

    if intent == "Schemes_Know_Intent":
        names = extract_scheme_names(generated_response)
        if names:
            st.session_state.scheme_names = [n for n in names]
            st.session_state.scheme_names_str = " ".join([f"{i+1}. {n}" for i, n in enumerate(names, 1)])
            logger.info(f"Stored scheme names from response: {st.session_state.scheme_names_str}")
    elif intent in {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent","Contextual_Follow_Up","Confirmation_New_RAG"} and referenced_scheme:
        if referenced_scheme not in st.session_state.scheme_names:
            st.session_state.scheme_names = [referenced_scheme]
            st.session_state.scheme_names_str = f"1. {referenced_scheme}"
        logger.info(f"Maintaining stored scheme names: {st.session_state.scheme_names_str}")

    step = time.time()
    hindi_audio_script = generate_hindi_audio_script(generated_response, user_info)
    record("audio_script", step)

    # Save conversation to MongoDB
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": generated_response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_audio_script},
        ]
        if not any(
            any(msg.get("interaction_id") == interaction_id for msg in conv["messages"])
            for conv in conversations
        ):
            step = time.time()
            data_manager.save_conversation(session_id, mobile_number, messages_to_save)
            record("save_conversation", step)
            logger.info(
                f"Saved conversation for session {session_id}: {query} -> {generated_response} (Interaction ID: {interaction_id})"
            )
        else:
            logger.debug(
                f"Skipped saving duplicate conversation for query: {query} (Interaction ID: {interaction_id})"
            )
    except Exception as e:
        logger.error(f"Failed to save conversation for session {session_id}: {str(e)}")

    log_timings()
    return generated_response, hindi_audio_script