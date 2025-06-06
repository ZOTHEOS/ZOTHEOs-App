# FILE: modules/user_auth.py

import json
import os
import sys 
import logging

logger = logging.getLogger("ZOTHEOS_UserAuth")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

USERS_FILE_PATH = "stripe_users.json" # Default name, path will be determined below

try:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        # _MEIPASS is the path to the temporary directory where PyInstaller unpacks files
        application_path = sys._MEIPASS
        logger.debug(f"UserAuth running in PyInstaller bundle. MEIPASS: {application_path}")
        # stripe_users.json should be in the root of _MEIPASS, alongside the .exe's main script
        candidate_path = os.path.join(application_path, "stripe_users.json")
        # Check if it exists there directly
        if os.path.exists(candidate_path):
             USERS_FILE_PATH = candidate_path
        else:
            # Fallback: if user_auth.py is in a 'modules' subdir within _MEIPASS, try one level up for stripe_users.json
            # This handles if --add-data "modules;modules" was used and user_auth.py is at _MEIPASS/modules/user_auth.py
            # and stripe_users.json was added to _MEIPASS/.
            # This might happen if the main script is also at the root of _MEIPASS.
            current_module_dir = os.path.dirname(os.path.abspath(__file__)) # This would be _MEIPASS/modules
            project_root_guess = os.path.dirname(current_module_dir) # This would be _MEIPASS
            alt_candidate_path = os.path.join(project_root_guess, "stripe_users.json")
            if os.path.exists(alt_candidate_path):
                USERS_FILE_PATH = alt_candidate_path
            else:
                # If still not found, log a warning. The get_user_tier will handle the FileNotFoundError.
                 logger.warning(f"stripe_users.json not found at primary bundle path '{candidate_path}' or alternate '{alt_candidate_path}'. It must be bundled at the root relative to the main script.")
                 # Keep USERS_FILE_PATH as "stripe_users.json" to allow relative loading if script is at root of _MEIPASS
                 # This will likely fail if not at root, and get_user_tier will default to 'free'
    else:
        # Running as a normal script (user_auth.py is in modules folder)
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.dirname(script_dir) 
        USERS_FILE_PATH = os.path.join(project_root, "stripe_users.json")
    
    logger.info(f"User authentication file path dynamically set to: {os.path.abspath(USERS_FILE_PATH) if os.path.isabs(USERS_FILE_PATH) else USERS_FILE_PATH}")

except Exception as e_path:
    logger.error(f"Critical error setting USERS_FILE_PATH in user_auth.py: {e_path}. Using default 'stripe_users.json'.")
    USERS_FILE_PATH = "stripe_users.json"


def get_user_tier(user_token: str) -> str:
    """
    Retrieves the user's tier based on their token from stripe_users.json.
    Defaults to 'free' if token not found or an error occurs.
    """
    default_tier = "free"
    if not user_token or not isinstance(user_token, str) or not user_token.strip():
        # This log is fine, but can be noisy if called frequently with no token (e.g. UI init)
        # logger.debug("No user token provided or invalid token, defaulting to 'free' tier.")
        return default_tier

    try:
        # Log the path being attempted for diagnosis
        logger.debug(f"Attempting to load users file from: {os.path.abspath(USERS_FILE_PATH)}")
        if not os.path.exists(USERS_FILE_PATH):
            logger.warning(f"Users file '{USERS_FILE_PATH}' not found. Defaulting all users to '{default_tier}' tier. Please ensure this file is correctly bundled at the application root.")
            return default_tier

        with open(USERS_FILE_PATH, "r", encoding="utf-8") as f:
            users = json.load(f)
        
        tier = users.get(user_token, default_tier)
        log_token_display = user_token[:3] + "***" if len(user_token) > 3 else user_token
        logger.info(f"Token '{log_token_display}' resolved to tier: '{tier}'.")
        return tier
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {USERS_FILE_PATH}. Defaulting to '{default_tier}' tier.", exc_info=True)
        return default_tier
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_user_tier: {e}. Defaulting to '{default_tier}' tier.", exc_info=True)
        return default_tier

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG) 
    print(f"--- Testing user_auth.py ---")
    print(f"USERS_FILE_PATH is configured as: {os.path.abspath(USERS_FILE_PATH)}")

    # Create a dummy stripe_users.json in the expected location for testing if it doesn't exist
    # For standalone testing, this means it looks for it relative to where this script would be if it were project root.
    # If this script is in 'modules', project_root is one level up.
    
    # Determine the correct test path based on whether script is run directly or from project root
    if os.path.basename(os.getcwd()) == "modules": # If CWD is modules folder
        test_stripe_users_path = "../stripe_users.json"
    else: # If CWD is project root (where zotheos_interface_public.py is)
        test_stripe_users_path = "stripe_users.json"

    if not os.path.exists(test_stripe_users_path):
        print(f"'{test_stripe_users_path}' not found relative to CWD ({os.getcwd()}). Creating a dummy file for testing purposes at this location.")
        dummy_users_for_test = {
            "TOKEN_FREE_USER": "free",
            "TOKEN_STARTER_USER": "starter",
            "TOKEN_PRO_USER": "pro"
        }
        try:
            with open(test_stripe_users_path, "w") as f:
                json.dump(dummy_users_for_test, f, indent=2)
            print(f"Dummy '{test_stripe_users_path}' created successfully with test tokens.")
            # Update USERS_FILE_PATH for this test run if it was different
            USERS_FILE_PATH = test_stripe_users_path 
            print(f"For this test run, USERS_FILE_PATH is now: {os.path.abspath(USERS_FILE_PATH)}")
        except Exception as e_create:
            print(f"Could not create dummy users file at {test_stripe_users_path}: {e_create}")
            print("Please ensure stripe_users.json exists in the project root for user_auth.py to function correctly.")
    
    print("\nTest Results:")
    test_tokens = {
        "Pro User Token": "TOKEN_PRO_USER",
        "Starter User Token": "TOKEN_STARTER_USER",
        "Free User Token": "TOKEN_FREE_USER",
        "Non-existent Token": "invalid_dummy_token_123",
        "Empty String Token": "",
    }

    for description, token_to_test in test_tokens.items():
        actual_tier = get_user_tier(token_to_test)
        print(f"- Test '{description}' (Input: '{token_to_test}'): Tier = '{actual_tier}'")

    print(f"\n--- End of user_auth.py test ---")