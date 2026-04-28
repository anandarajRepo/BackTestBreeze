import argparse
import os
import sys
import urllib.parse

from dotenv import load_dotenv, set_key

load_dotenv()

ENV_FILE = ".env"


def _validate_auth_credentials():
    api_key = os.getenv("BREEZE_API_KEY")
    api_secret = os.getenv("BREEZE_API_SECRET")

    if not api_key:
        print("Error: BREEZE_API_KEY is not set in your .env file.")
        print("Get your API key from the ICICI Direct API portal: https://api.icicidirect.com/")
        sys.exit(1)

    if not api_secret:
        print("Error: BREEZE_API_SECRET is not set in your .env file.")
        print("Get your API secret from the ICICI Direct API portal: https://api.icicidirect.com/")
        sys.exit(1)

    return api_key, api_secret


def _verify_token(breeze):
    try:
        response = breeze.get_customer_details()
        if response and response.get("Status") == 200:
            customer = response.get("Success", {})
            user_id = customer.get("idirect_userid", "Unknown") if customer else "Unknown"
            print(f"Token verified successfully. Logged in as: {user_id}")
            return True
        else:
            print(f"Token verification failed: {response}")
            return False
    except Exception as e:
        print(f"Error verifying token: {e}")
        return False


def run_auth():
    api_key, api_secret = _validate_auth_credentials()

    try:
        from breeze_connect import BreezeConnect
    except ImportError:
        print("Error: breeze-connect library is not installed.")
        print("Install it with: pip install breeze-connect")
        sys.exit(1)

    breeze = BreezeConnect(api_key=api_key)

    login_url = (
        "https://api.icicidirect.com/apiuser/login"
        f"?api_key={urllib.parse.quote_plus(api_key)}"
    )

    print("\n=== ICICI Direct Breeze Authentication ===")
    print("\nStep 1: Open the following URL in your browser to log in:")
    print(f"\n  {login_url}\n")
    print("Step 2: After logging in, you will be redirected to a URL.")
    print("        Copy the value of the 'skey' parameter from the redirect URL.")
    print("        Example redirect: https://example.com/?skey=YOUR_SESSION_TOKEN\n")

    session_token = input("Step 3: Paste the session token (skey value) here: ").strip()

    if not session_token:
        print("Error: No session token provided.")
        sys.exit(1)

    print("\nGenerating session...")
    try:
        response = breeze.generate_session(
            api_secret=api_secret,
            session_token=session_token,
        )
        if response and response.get("Status") != 200:
            print(f"Error generating session: {response.get('Error', response)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error generating session: {e}")
        sys.exit(1)

    set_key(ENV_FILE, "BREEZE_SESSION_TOKEN", session_token)
    masked = f"***{session_token[-4:]}" if len(session_token) >= 4 else "****"
    print(f"\nSession token saved to {ENV_FILE}")
    print(f"BREEZE_SESSION_TOKEN={masked}")

    print("\nVerifying token...")
    _verify_token(breeze)


def main():
    parser = argparse.ArgumentParser(
        description="BackTestBreeze CLI - ICICI Direct Breeze API"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("auth", help="Generate and save a Breeze API session token")

    args = parser.parse_args()

    if args.command == "auth":
        run_auth()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
