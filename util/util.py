import re

def extract_filename_from_prompt(prompt: str) -> str:
    """
    Extracts a filename from a prompt string, ignoring email addresses.

    The function looks for a pattern starting with '@' that is not preceded by
    a word character (to avoid matching email addresses). If a match is found,
    the filename is extracted and any trailing punctuation (.,!?) is removed.

    Args:
        prompt: The input string from which to extract the filename.

    Returns:
        The cleaned filename as a string, or an empty string if no valid match is found.
    """
    # Negative lookbehind (?<!\w) ensures the @ is not preceded by a word character (part of an email)
    match = re.search(r"(?<!\w)@([^,\s]+)", prompt)
    if match:
        # Strip common trailing punctuation from the extracted filename
        filename = match.group(1)
        return filename.rstrip('.,!?').lower()
    return ""

def normalize_filename(filename: str) -> str:
    """
    Normalizes a filename by replacing whitespace with underscores and converting to lowercase.

    Args:
        filename: The input filename string.

    Returns:
        A normalized filename string.
    """
    # Replace any whitespace character with an underscore
    no_whitespace = re.sub(r"\s+", "_", filename)
    # Convert to lowercase
    return no_whitespace.lower()
