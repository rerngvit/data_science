import string
class TextPreprocessor():
    def __init__(self):
        pass

    ## Text cleaning utility
    def replacing_abbrev_with_full_word(self, text):
        cur_text = text
        abbrev_replace_dict = {
            "'s": " is ",
            "'re": " are ",
            "can't": " can not ",
            "don't": " do not ",
            "couldn't": " could not ",
            "wouldn't": " would not ",
            "didn't": " did not ",
            "hasn't": " has not ",
            "haven't": " have not ",
            "won't": " will not"
        }

        for k, v in abbrev_replace_dict.items():
            cur_text = cur_text.replace(k, v)
        return cur_text

    # remove puncation
    def remove_punctuations(self, text):
        cur_text = text
        for punct in string.punctuation:
            cur_text = cur_text.replace(punct, "")
        return cur_text

    def remove_numbers(self, text):
        def token_is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        return " ".join([i for i in text.split(" ") if not token_is_int(i)])

    def remove_empty_tokens(self, text):
        return " ".join([i for i in text.split(" ") if i != ""])

    def preprocess_text(self, msg_text):
        cur_text = msg_text
        cur_text = self.replacing_abbrev_with_full_word(cur_text)
        cur_text = self.remove_punctuations(cur_text)
        cur_text = self.remove_numbers(cur_text)
        cur_text = self.remove_empty_tokens(cur_text)
        cur_text = cur_text.lower().strip()  # Normalize to lower case
        return cur_text