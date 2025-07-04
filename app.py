import os
import streamlit as st
import random
import string
from datetime import datetime, timedelta
from msme_bot import (
    load_rag_data,
    load_dfl_data,
    process_query,
    welcome_user,
    detect_language,
)
from data import DataManager, STATE_MAPPING
import numpy as np
import logging
from tts import synthesize, audio_player
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize DataManager
data_manager = DataManager()

# Initialize Streamlit session state
if "scheme_vector_store" not in st.session_state:
    st.session_state.scheme_vector_store = load_rag_data()
if "dfl_vector_store" not in st.session_state:
    st.session_state.dfl_vector_store = load_dfl_data()

if "page" not in st.session_state:
    st.session_state.page = "login"

if "user" not in st.session_state:
    st.session_state.user = None

if "otp" not in st.session_state:
    st.session_state.otp = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "otp_generated" not in st.session_state:
    st.session_state.otp_generated = False

if "welcome_message_sent" not in st.session_state:
    st.session_state.welcome_message_sent = False

if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None

if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}
if "dfl_rag_cache" not in st.session_state:
    st.session_state.dfl_rag_cache = {}
if "scheme_flow_active" not in st.session_state:
    st.session_state.scheme_flow_active = False
if "scheme_flow_step" not in st.session_state:
    st.session_state.scheme_flow_step = None
if "scheme_flow_data" not in st.session_state:
    st.session_state.scheme_flow_data = {}
if "scheme_names" not in st.session_state:
    st.session_state.scheme_names = []
if "scheme_names_str" not in st.session_state:
    st.session_state.scheme_names_str = ""

# Generate session ID
def generate_session_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

# Generate unique query ID
def generate_query_id(query, timestamp):
    return f"{query[:50]}_{timestamp.strftime('%Y%m%d%H%M%S')}"

# Restore session from URL query parameters
def restore_session_from_url():
    query_params = st.query_params
    session_id = query_params.get("session_id")
    if session_id and not st.session_state.session_id:
        # Check if session_id exists in MongoDB and is active
        session = data_manager.db.sessions.find_one({"session_id": session_id, "end_time": {"$exists": False}})
        if session:
            mobile_number = session.get("mobile_number")
            user = data_manager.find_user(mobile_number)
            if user:
                st.session_state.session_id = session_id
                st.session_state.user = {
                    "fname": user.get("fname"),
                    "lname": user.get("lname"),
                    "mobile_number": user.get("mobile_number"),
                    "state_id": user.get("state_id", "Unknown"),
                    "state_name": user.get("state_name", "Unknown"),
                    "business_name": user.get("business_name"),
                    "business_category": user.get("business_category"),
                    "language": user.get("language", "English"),
                    "gender": user.get("gender"),
                }
                st.session_state.page = "chat"
                # Restore messages from MongoDB, but do NOT include audio_script
                conversations = data_manager.get_conversations(mobile_number, limit=10)
                all_messages = []
                for conv in conversations:
                    for msg in conv["messages"]:
                        if "content" not in msg or "role" not in msg or "timestamp" not in msg:
                            logger.warning(f"Skipping malformed message in MongoDB: {msg}")
                            continue
                        all_messages.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "timestamp": msg["timestamp"],
                            # Do NOT retrieve audio_script
                        })
                st.session_state.messages = all_messages
                logger.info(f"Restored session {session_id} for user {mobile_number}")
                return True
    return False



# token authentication function
def token_authentication():
    query_params = st.query_params
    token = query_params.get("token", [None])
    if token:
        BASE_URL = os.getenv("HQ_API_URL", "https://customer-admin-test.haqdarshak.com")
        ENDPOINT = "/person/get/citizen-details"
        API_URL = f"{BASE_URL}{ENDPOINT}"
        
        try:
            response = requests.get(API_URL, headers={"Authorization": f"Bearer {token}"})
            if response.status_code == 200:
                data = response.json()

                if data.get("responseCode") == "OK" and data["params"]["status"] == "successful":
                    result = data["result"]
                    st.session_state.user = {
                        "fname": result.get("firstName", ""),
                        "lname": result.get("lastName", ""),
                        "mobile_number": result.get("contactNumber", ""),
                        "gender": result.get("gender", ""),
                        "state_name": result.get("state", ""),
                        "dob": result.get("dob", ""),
                        "pincode": result.get("pincode", ""),
                        "business_name": result.get("bussinessName", ""),
                        "business_category": result.get("employmentType", ""),
                        "language": "English",
                        "state_id": STATE_MAPPING.get(result.get("state", ""), "Unknown")
                    }

                    # Generate session_id
                    if not st.session_state.session_id:
                        st.session_state.session_id = generate_session_id()
                        data_manager.start_session(
                            st.session_state.user["mobile_number"],
                            st.session_state.session_id,
                            st.session_state.user
                        )

                    st.session_state.messages = []
                    st.session_state.page = "chat"
                    st.session_state.welcome_message_sent = False
                    st.query_params["session_id"] = st.session_state.session_id
                    st.success("Login successful via token!")
                    st.rerun()
                    st.session_state.page = "chat"
                    return True
                else:
                    st.error("Token is invalid or user details not found.")
            else:
                st.error(f"API call failed: {response.status_code}")
        except Exception as e:
            st.error("Error contacting citizen API.")
            logger.exception("Citizen API call failed")
        return

    # Try to restore session from URL
    if restore_session_from_url():
        st.rerun()
        return

    st.title("Login")
    st.markdown("Enter your mobile number to log in.")

    mobile_number = st.text_input("Mobile Number (10 digits)")
    if st.button("Generate OTP"):
        if not mobile_number.isdigit() or len(mobile_number) != 10:
            st.error("Mobile number must be 10 digits.")
        else:
            user = data_manager.find_user(mobile_number)
            if not user:
                st.error("Mobile number not registered. Please register first.")
            else:
                st.session_state.otp = str(random.randint(100000, 999999))
                st.session_state.temp_mobile = mobile_number
                st.session_state.otp_generated = True
                st.info(f"Simulated OTP sent: {st.session_state.otp}")

    if st.session_state.otp_generated:
        otp_input = st.text_input("Enter OTP", type="password")
        if st.button("Verify OTP"):
            if otp_input == st.session_state.otp:
                user = data_manager.find_user(st.session_state.temp_mobile)
                # Store user data including state_id, state_name, and language in session state
                st.session_state.user = {
                    "fname": user.get("fname"),
                    "lname": user.get("lname"),
                    "mobile_number": user.get("mobile_number"),
                    "state_id": user.get("state_id", "Unknown"),
                    "state_name": user.get("state_name", "Unknown"),
                    "business_name": user.get("business_name"),
                    "business_category": user.get("business_category"),
                    "language": user.get("language", "English"),
                    "gender": user.get("gender"),
                }
                # Generate session_id only if not already set
                if not st.session_state.session_id:
                    st.session_state.session_id = generate_session_id()
                    # Always pass user_data, as it's optional in start_session
                    logger.debug(f"Calling start_session with mobile: {st.session_state.temp_mobile}, session_id: {st.session_state.session_id}, user_data: {st.session_state.user}")
                    data_manager.start_session(st.session_state.temp_mobile, st.session_state.session_id, st.session_state.user)
                st.session_state.messages = [] # Clear messages on successful login
                st.session_state.page = "chat"
                st.session_state.otp_generated = False
                st.session_state.otp = None
                st.session_state.last_query_id = None
                st.session_state.welcome_message_sent = False # Ensure welcome message is sent on fresh login
                # Add session_id to URL
                st.query_params["session_id"] = st.session_state.session_id
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid OTP. Please try again.")

# Chat page
def chat_page():
    # Verify session validity
    if not st.session_state.session_id or not st.session_state.user:
        # Try to restore session from URL
        if restore_session_from_url():
            st.rerun()
            return
        # If restoration fails, redirect to login
        st.session_state.page = "login"
        st.session_state.user = None
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.otp_generated = False
        st.session_state.welcome_message_sent = False
        st.session_state.last_query_id = None
        st.session_state.scheme_flow_active = False
        st.session_state.scheme_flow_step = None
        st.session_state.scheme_flow_data = {}
        st.query_params.clear()
        st.rerun()
        return

    # Check MongoDB session validity
    session = data_manager.db.sessions.find_one({"session_id": st.session_state.session_id, "end_time": {"$exists": False}})
    if not session:
        st.session_state.page = "login"
        st.session_state.user = None
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.otp_generated = False
        st.session_state.welcome_message_sent = False
        st.session_state.last_query_id = None
        st.session_state.scheme_flow_active = False
        st.session_state.scheme_flow_step = None
        st.session_state.scheme_flow_data = {}
        st.query_params.clear()
        st.rerun()
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"Welcome, {st.session_state.user['fname']}")            

    # Ensure session_id is in URL
    if "session_id" not in st.query_params or st.query_params["session_id"] != st.session_state.session_id:
        st.query_params["session_id"] = st.session_state.session_id

    # Trigger welcome message only for new users or on fresh login
    if not st.session_state.welcome_message_sent:
        conversations = data_manager.get_conversations(
            st.session_state.user["mobile_number"], limit=10
        )
        user_type = "returning" if conversations else "new"

        if user_type == "new":
            welcome_response, welcome_audio_script = process_query(
                "welcome",
                st.session_state.scheme_vector_store,
                st.session_state.dfl_vector_store,
                st.session_state.session_id,
                st.session_state.user["mobile_number"],
                user_language=st.session_state.user["language"]
            )
            if welcome_response:  # Only append if a welcome message was generated
                if welcome_audio_script:
                    audio_bytes = synthesize(welcome_audio_script, "Hindi")
                    audio_player(audio_bytes, autoplay=True)
                st.session_state.welcome_message_sent = True


    # Combine past conversations from MongoDB and current session messages
    st.subheader("Conversation History")
    all_messages = []

    # Fetch past conversations from MongoDB - DO NOT retrieve audio_script
    conversations = data_manager.get_conversations(st.session_state.user["mobile_number"], limit=10)
    for conv in conversations:
        for msg in conv["messages"]:
            if "content" not in msg or "role" not in msg or "timestamp" not in msg:
                logger.warning(f"Skipping malformed message in MongoDB: {msg}")
                continue
            all_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            })

    # Add current session messages - DO NOT include audio_script when adding to all_messages
    for msg in st.session_state.messages:
        # Check for content, role, and timestamp for uniqueness
        if not any(m["role"] == msg["role"] and m["content"] == msg["content"] for m in all_messages):
            all_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            })
            logger.debug(f"Added session message to all_messages: {msg['role']} - {msg['content']} ({msg['timestamp']})")

    # Sort all messages by timestamp
    all_messages.sort(key=lambda x: x["timestamp"])

    # Display all messages. Audio player is only for the latest response in the chat input.
    for msg in all_messages:
        with st.chat_message(msg["role"], avatar="logo.jpeg" if msg["role"] == "assistant" else None):
            if msg["role"] == "user":
                st.markdown(f"{msg['content']} *({msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})*")
            else: # Assistant messages
                # --- MODIFICATION START ---
                full_content = msg["content"]
                display_timestamp = msg["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
                expand_threshold = 200 

                if len(full_content) > expand_threshold:
                    # Create the expander label including the timestamp
                    concise_preview = full_content[:expand_threshold].rsplit(' ', 1)[0] + "..."
                    expander_label = f"{concise_preview} *({display_timestamp})*"
                    
                    # Place the expander. The full content goes inside the expander.
                    with st.expander(expander_label):
                        st.markdown(full_content)
                else:
                    # If content is short, display it directly with timestamp.
                    st.markdown(f"{full_content} *({display_timestamp})*")
                # --- MODIFICATION END ---


    # Chat input
    query = st.chat_input("Type your query here...")
    if query:
        # Generate a unique query ID to prevent double-processing
        query_timestamp = datetime.utcnow()
        query_id = generate_query_id(query, query_timestamp)
        if query_id != st.session_state.last_query_id:
            st.session_state.last_query_id = query_id
            # Append user query if not already in session state
            last_msg = st.session_state.messages[-1] if st.session_state.messages else None
            if not last_msg or not (last_msg["role"] == "user" and last_msg["content"] == query):
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": query_timestamp
                })
                with st.chat_message("user"):
                    st.markdown(f"{query} *({query_timestamp.strftime('%Y-%m-%d %H:%M:%S')})*")
                logger.debug(f"Appended user query to session state: {query} (ID: {query_id})")

            # Display typing indicator while generating response
            with st.chat_message("assistant", avatar="logo.jpeg"):
                with st.spinner("Assistant is typing..."):
                    response, audio_script_for_tts = process_query(
                        query,
                        st.session_state.scheme_vector_store,
                        st.session_state.dfl_vector_store,
                        st.session_state.session_id,
                        st.session_state.user["mobile_number"],
                        user_language=st.session_state.user["language"]
                    )
                response_timestamp = datetime.utcnow()

                # Play audio only for the NEW response
                if audio_script_for_tts:
                    audio_bytes = synthesize(audio_script_for_tts, "Hindi")
                    audio_player(audio_bytes, autoplay=True)
                
                # --- MODIFICATION START ---
                full_content_response = response
                display_timestamp_response = response_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                expand_threshold_response = 200 

                if len(full_content_response) > expand_threshold_response:
                    concise_preview_response = full_content_response[:expand_threshold_response].rsplit(' ', 1)[0] + "..."
                    expander_label_response = f"{concise_preview_response} *({display_timestamp_response})*"
                    
                    with st.expander(expander_label_response):
                        st.markdown(full_content_response)
                else:
                    st.markdown(f"{full_content_response} *({display_timestamp_response})*")
                # --- MODIFICATION END ---

            last_msg = st.session_state.messages[-1] if st.session_state.messages else None
            if not last_msg or not (last_msg["role"] == "assistant" and last_msg["content"] == response):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": response_timestamp,
                })
                logger.debug(f"Appended bot response to session state: {response} (Query ID: {query_id})")
                logger.debug("Bot response appended")

# Main app logic
# Check for session restoration first
restore_session_from_url()

if not st.session_state.get("user"):
    token_authentication()

# if st.session_state.page == "login":
#     token_authentication()
# elif st.session_state.page == "register":
#     registration_page()
if st.session_state.page == "chat":
    chat_page()

# Link to registration
# if st.session_state.page == "login":
#     if st.button("New User? Register Here"):
#         st.session_state.page = "register"
#         st.query_params.clear()
#         st.rerun()

# Update existing users to include state_id and state_name
data_manager.update_existing_users_state()