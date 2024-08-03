import os
import random
import string


# FUNCTIONS
def clear_terminal():
    # Check the operating system
    if os.name == "posix":  # For Linux and macOS
        os.system("clear")


def debug(*texts):
    texts = list(texts)

    display_text = "\n".join([str(text) for text in texts])
    print(display_text)
    x = input(" ")

    if x == "x":
        clear_terminal()

        file_path = "logs.log"
        with open(file_path, "w"):
            pass

        exit()


def random_suffix(char_len: int = 8):
    """
    Generate a random suffix
    """
    # using random.choices()
    # generating random strings
    return "".join(random.choices(string.ascii_letters + string.digits, k=char_len))


if __name__ == "__main__":
    print("done.")
