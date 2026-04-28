from breeze_connect import BreezeConnect
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize SDK
breeze = BreezeConnect(api_key=os.getenv("BREEZE_API_KEY"))

# Generate Session
# BREEZE_API_KEY: Your API Key from the ICICI Breeze developer portal
# BREEZE_API_SECRET: Your API Secret from the ICICI Breeze developer portal
# BREEZE_SESSION_TOKEN: The session token from the login redirect URL
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("BREEZE_SESSION_TOKEN")
)

# You can now use 'breeze' object for trading
print("Session Generated Successfully")