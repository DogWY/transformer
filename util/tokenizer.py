import spacy

class Tokenizer:
    def __init__(self):
        """
        加载spacy中的指定分词器
        """
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        
    def tokenize_de(self, text):
        """
        使用spacy提供的分词器对德语文本进行分词
        """
        return [token.text for token in self.spacy_de.tokenizer(text)]
    
    def tokenize_en(self, text):
        """
        使用spacy提供的分词器对英语文本进行分词
        """
        return [token.text for token in self.spacy_en.tokenizer(text)]