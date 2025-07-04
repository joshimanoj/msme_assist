import os
import logging
import time
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import quote_plus, urlparse, urlunparse

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# State mapping
STATE_MAPPING = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CT": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OR": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "UT": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

@st.cache_resource
def get_mongo_client():
    """Initialize and cache MongoDB client."""
    logger.info("Initializing MongoDB client")
    start_time = time.time()
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    
    try:
        # Parse the MongoDB URI to extract username and password
        parsed_uri = urlparse(mongo_uri)
        username = parsed_uri.username
        password = parsed_uri.password
        
        # URL-encode username and password if they exist
        if username and password:
            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            
            # Reconstruct the URI with encoded credentials
            netloc = f"{encoded_username}:{encoded_password}@{parsed_uri.hostname}"
            if parsed_uri.port:
                netloc += f":{parsed_uri.port}"
            # Rebuild the URI
            mongo_uri = urlunparse((
                parsed_uri.scheme,
                netloc,
                parsed_uri.path,
                parsed_uri.params,
                parsed_uri.query,
                parsed_uri.fragment
            ))
        
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True,
            maxPoolSize=50
        )
        # Test the connection
        client.server_info()
        logger.info(f"MongoDB client initialized in {time.time() - start_time:.2f} seconds at {mongo_uri}")
        return client
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB client initialization: {str(e)}")
        raise

class DataManager:
    """Manages MongoDB operations for users, sessions, conversations, and embeddings."""

    def __init__(self):
        self.client = get_mongo_client()
        db_name = os.getenv("MONGO_DB_NAME", "haqdarshak")
        self.db = self.client[db_name]
        logger.info(f"Using database {db_name}")
        self.update_existing_users_language()

    def register_user(self, fname, lname, mobile_number, state, business_name, business_category, language,
                      gender):
        """Register a new user and return success status with message."""
        if not all([fname, lname, mobile_number, state, business_name, business_category, language, gender]):
            logger.error("All registration fields must be non-empty")
            return False, "All fields are required."
        if not isinstance(mobile_number, str) or not mobile_number.isdigit():
            logger.error(f"Invalid mobile_number: {mobile_number} must be a string of digits")
            return False, "Mobile number must be a valid number."
        if not isinstance(state, str) or state not in STATE_MAPPING.values():
            logger.error(f"Invalid state: {state}")
            return False, "Invalid state selected."
        if language not in ["English", "Hindi"]:
            logger.error(f"Invalid language: {language}")
            return False, "Invalid language selected."
        if gender not in ["Male", "Female", "Other"]:
            logger.error(f"Invalid gender: {gender}")
            return False, "Invalid gender selected."

        try:
            existing_user = self.db.users.find_one({"mobile_number": mobile_number})
            if existing_user:
                logger.warning(f"Mobile number {mobile_number} already registered")
                return False, "Mobile number already registered. Please log in."
            
            state_id = None
            for key, value in STATE_MAPPING.items():
                if value == state:
                    state_id = key
                    break
            
            user_data = {
                "fname": fname,
                "lname": lname,
                "mobile_number": mobile_number,
                "state_id": state_id,
                "state_name": state,
                "business_name": business_name,
                "business_category": business_category,
                "language": language,
                "gender": gender,
                "created_at": datetime.utcnow()
            }
            self.db.users.insert_one(user_data)
            logger.info(f"Registered user with mobile {mobile_number}, state_id {state_id}, state_name {state}, language {language}")
            return True, "Registration successful! Please log in."
        except OperationFailure as e:
            logger.error(f"Operation failed while registering user with mobile {mobile_number}: {str(e)}")
            return False, f"Error registering user: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error while registering user with mobile {mobile_number}: {str(e)}")
            return False, f"Error registering user: {str(e)}"

    def find_user(self, mobile_number):
        """Find user by mobile number."""
        if not isinstance(mobile_number, str) or not mobile_number.isdigit():
            logger.error(f"Invalid mobile_number: {mobile_number} must be a string of digits")
            return None

        try:
            user = self.db.users.find_one({"mobile_number": mobile_number})
            if user:
                logger.info(f"Found user with mobile {mobile_number}")
            else:
                logger.warning(f"No user found with mobile {mobile_number}")
            return user
        except OperationFailure as e:
            logger.error(f"Operation failed while finding user with mobile {mobile_number}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while finding user with mobile {mobile_number}: {str(e)}")
            return None
        
    def update_user_profile(self, mobile_number, **fields):
        """Update user profile fields for the given mobile number."""
        if not isinstance(mobile_number, str) or not mobile_number.isdigit():
            logger.error(f"Invalid mobile_number: {mobile_number} must be a string of digits")
            return False, "Invalid mobile number."

        allowed_fields = {
            "fname", "lname", "business_name", "business_category", "language",
            "gender", "state"
        }

        update_data = {}
        for key, value in fields.items():
            if key not in allowed_fields:
                logger.warning(f"Ignoring invalid field for update: {key}")
                continue
            if key == "language" and value not in ["English", "Hindi"]:
                logger.error(f"Invalid language: {value}")
                return False, "Invalid language selected."
            if key == "gender" and value not in ["Male", "Female", "Other"]:
                logger.error(f"Invalid gender: {value}")
                return False, "Invalid gender selected."
            if key == "state":
                if value not in STATE_MAPPING.values():
                    logger.error(f"Invalid state: {value}")
                    return False, "Invalid state selected."
                update_data["state_name"] = value
                for k, v in STATE_MAPPING.items():
                    if v == value:
                        update_data["state_id"] = k
                        break
            else:
                update_data[key] = value

        if not update_data:
            logger.warning("No valid fields provided for update")
            return False, "No valid fields to update."

        try:
            result = self.db.users.update_one({"mobile_number": mobile_number}, {"$set": update_data})
            if result.matched_count == 0:
                logger.warning(f"No user found with mobile {mobile_number} to update")
                return False, "User not found."
            logger.info(f"Updated user {mobile_number} with fields {list(update_data.keys())}")
            return True, "Profile updated successfully."
        except OperationFailure as e:
            logger.error(f"Operation failed while updating user with mobile {mobile_number}: {str(e)}")
            return False, f"Error updating profile: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error while updating user with mobile {mobile_number}: {str(e)}")
            return False, f"Error updating profile: {str(e)}"


    def start_session(self, mobile_number, session_id, user_data=None):
        """Log session start, optionally storing user_data."""
        if not session_id or not isinstance(session_id, str):
            logger.error("Invalid session_id: session_id must be a non-empty string")
            raise ValueError("session_id must be a non-empty string")
        if not mobile_number or not isinstance(mobile_number, str):
            logger.error("Invalid mobile_number: mobile_number must be a non-empty string")
            raise ValueError("mobile_number must be a non-empty string")

        try:
            session_data = {
                "session_id": session_id,
                "mobile_number": mobile_number,
                "start_time": datetime.utcnow()
            }
            if user_data:
                session_data["user_data"] = user_data
            self.db.sessions.insert_one(session_data)
            logger.info(f"Started session {session_id} for mobile {mobile_number} with user_data: {user_data}")
        except OperationFailure as e:
            logger.error(f"Operation failed while starting session {session_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while starting session {session_id}: {str(e)}")
            raise

    def end_session(self, session_id):
        """Log session end."""
        if not session_id or not isinstance(session_id, str):
            logger.error("Invalid session_id: session_id must be a non-empty string")
            raise ValueError("session_id must be a non-empty string")

        try:
            result = self.db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"end_time": datetime.utcnow()}}
            )
            if result.matched_count == 0:
                logger.warning(f"No session found with session_id {session_id} to end")
            else:
                logger.info(f"Ended session {session_id}")
        except OperationFailure as e:
            logger.error(f"Operation failed while ending session {session_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while ending session {session_id}: {str(e)}")
            raise

    def save_conversation(self, session_id, mobile_number, messages):
        """Save or append messages to a conversation in MongoDB."""
        if not session_id or not isinstance(session_id, str):
            logger.error("Invalid session_id: session_id must be a non-empty string")
            raise ValueError("session_id must be a non-empty string")
        if not mobile_number or not isinstance(mobile_number, str):
            logger.error("Invalid mobile_number: mobile_number must be a non-empty string")
            raise ValueError("mobile_number must be a non-empty string")
        if not messages or not isinstance(messages, list) or not all(isinstance(msg, dict) for msg in messages):
            logger.error("Invalid messages: messages must be a non-empty list of dictionaries")
            raise ValueError("messages must be a non-empty list of dictionaries")

        try:
            existing_conversation = self.db.conversations.find_one({"session_id": session_id})
            if existing_conversation:
                self.db.conversations.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"messages": {"$each": messages}},
                        "$set": {"timestamp": datetime.utcnow()}
                    }
                )
                logger.info(f"Appended {len(messages)} messages to existing conversation for session {session_id}")
            else:
                conversation_data = {
                    "session_id": session_id,
                    "mobile_number": mobile_number,
                    "messages": messages,
                    "timestamp": datetime.utcnow()
                }
                self.db.conversations.insert_one(conversation_data)
                logger.info(f"Created new conversation for session {session_id} with mobile_number {mobile_number}")
        except OperationFailure as e:
            logger.error(f"Operation failed while saving conversation for session {session_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving conversation for session {session_id}: {str(e)}")
            raise

    def get_conversations(self, mobile_number, limit=None):
        """Retrieve conversations for a user sorted by most recent first.

        Parameters
        ----------
        mobile_number : str
            The mobile number of the user.
        limit : int, optional
            Maximum number of conversations to return. If ``None`` all
            conversations are returned.
        """
        if not mobile_number or not isinstance(mobile_number, str):
            logger.error("Invalid mobile_number: mobile_number must be a non-empty string")
            raise ValueError("mobile_number must be a non-empty string")

        try:
            cursor = self.db.conversations.find({"mobile_number": mobile_number}).sort("timestamp", -1)
            if isinstance(limit, int) and limit > 0:
                cursor = cursor.limit(limit)
            conversations = list(cursor)
            if not conversations:
                logger.warning(f"No conversations found for mobile_number {mobile_number}")
            else:
                logger.info(f"Retrieved {len(conversations)} conversations for mobile_number {mobile_number}")
            return conversations
        except OperationFailure as e:
            logger.error(f"Operation failed while retrieving conversations for mobile_number {mobile_number}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while retrieving conversations for mobile_number {mobile_number}: {str(e)}")
            return []

    def clear_conversations(self, mobile_number):
        """Delete all conversations for a user."""
        if not mobile_number or not isinstance(mobile_number, str):
            logger.error("Invalid mobile_number: mobile_number must be a non-empty string")
            raise ValueError("mobile_number must be a non-empty string")

        try:
            result = self.db.conversations.delete_many({"mobile_number": mobile_number})
            logger.info(f"Cleared {result.deleted_count} conversations for mobile {mobile_number}")
        except OperationFailure as e:
            logger.error(f"Operation failed while clearing conversations for mobile {mobile_number}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while clearing conversations for mobile {mobile_number}: {str(e)}")
            raise

    def save_embeddings(self, embeddings_data):
        """Save embeddings to MongoDB with a last_updated timestamp."""
        if not embeddings_data or not isinstance(embeddings_data, list) or not all(isinstance(item, dict) for item in embeddings_data):
            logger.error("Invalid embeddings_data: embeddings_data must be a non-empty list of dictionaries")
            raise ValueError("embeddings_data must be a non-empty list of dictionaries")

        try:
            self.db.embeddings.delete_many({})
            self.db.embeddings.insert_many(embeddings_data)
            self.db.embeddings_metadata.delete_many({})
            self.db.embeddings_metadata.insert_one({"last_updated": datetime.utcnow()})
            logger.info(f"Saved {len(embeddings_data)} embeddings to MongoDB")
        except OperationFailure as e:
            logger.error(f"Operation failed while saving embeddings: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving embeddings: {str(e)}")
            raise

    def get_embeddings(self):
        """Retrieve embeddings from MongoDB."""
        try:
            embeddings = list(self.db.embeddings.find())
            if not embeddings:
                logger.warning("No embeddings found in the database")
            else:
                logger.info(f"Retrieved {len(embeddings)} embeddings from MongoDB")
            return embeddings
        except OperationFailure as e:
            logger.error(f"Operation failed while retrieving embeddings: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while retrieving embeddings: {str(e)}")
            return []

    def get_last_updated(self):
        """Retrieve the last_updated timestamp for embeddings."""
        try:
            metadata = self.db.embeddings_metadata.find_one()
            if metadata and "last_updated" in metadata:
                logger.info(f"Last updated timestamp for embeddings: {metadata['last_updated']}")
                return metadata["last_updated"]
            logger.info("No last_updated timestamp found for embeddings")
            return None
        except OperationFailure as e:
            logger.error(f"Operation failed while retrieving last_updated timestamp: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while retrieving last_updated timestamp: {str(e)}")
            return None

    def update_existing_users_state(self):
        """Update existing users to add state_id 'MH' and state_name 'Maharashtra' if not present."""
        try:
            result = self.db.users.update_many(
                {"$or": [{"state_id": {"$exists": False}}, {"state_name": {"$exists": False}}]},
                {"$set": {"state_id": "MH", "state_name": "Maharashtra"}}
            )
            logger.info(f"Updated {result.modified_count} users with state_id 'MH' and state_name 'Maharashtra'")
        except OperationFailure as e:
            logger.error(f"Operation failed while updating users' state fields: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while updating users' state fields: {str(e)}")
            raise

    def update_existing_users_language(self):
        """Update existing users to add language 'English' if not present."""
        try:
            result = self.db.users.update_many(
                {"language": {"$exists": False}},
                {"$set": {"language": "English"}}
            )
            logger.info(f"Updated {result.modified_count} users with language 'English'")
        except OperationFailure as e:
            logger.error(f"Operation failed while updating users' language fields: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while updating users' language fields: {str(e)}")
            raise

    def close(self):
        """Close the MongoDB client connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("MongoDB client connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB client connection: {str(e)}")