from breeze_connect import BreezeConnect
import os

# Initialize SDK
breeze = BreezeConnect(api_key="YOUR_API_KEY")

# Generate Session
# app_key: Your API Key
# secret_key: Your API Secret
# session_token: The session key from the URL (manual step)
breeze.generate_session(
    api_secret=os.getenv("BREEZE_API_SECRET"),
    session_token=os.getenv("YOUR_SESSION_KEY_FROM_URL")
)

# You can now use 'breeze' object for trading
print("Session Generated Successfully")