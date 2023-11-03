from enchant.tokenize import Filter

class AcronymFilter(Filter):
    """If a word contains capital letters on non-first positions, ignore it"""

    def _skip(self, word):
        return word[1:] != word[1:].lower()

class LibFilter(Filter):
    """If a word starts with `lib` its probably a library name, so ignore it """

    def _skip(self, word):
        return word.lower().startswith("lib")
    
class TechFilter(Filter):
    def _skip(self, word):
        return '.' in word or '_' in word