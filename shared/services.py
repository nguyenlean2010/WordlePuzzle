import requests
import json

API_URL = "https://wordle.votee.dev:8000"

class GuessResult:
    def __init__(self, slot, guess, result):
        self.slot = slot
        self.guess = guess
        self.result = result

class WordleService:
    def guess_word(self, secret_word, guess, size=5):
        """Makes a guess against the Wordle API."""
        if secret_word is None:
            url = f"{API_URL}/daily?guess={guess}&size={size}"
        else:
            url = f"{API_URL}/word/{secret_word}?guess={guess}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            results = [GuessResult(item['slot'], item['guess'], item['result']) for item in response.json()]
            return results
        except requests.exceptions.RequestException as e:
            print(f"Error making guess: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None