#for logistic regression,naive bayes and svm
cname=['target','id','date','flag','user','text']
import pandas as pd
df=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1',names=cname)
import re
import string,time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
def remove_html(text):
    clean_text=re.sub(r'<.*?>','',text)
    return clean_text
def remove_url(text):
    clean_text=re.sub(r"http?://\S+|www\.\S+|\b\w+\.(com|org|net|io|gov|edu)\b\S*",'',text)
    return clean_text
  #CONVER TO LOWER CASE
def convert_to_lowercase(text):
      return text.lower()
  #REMOVE TWITTER TAGS
def remove_twitter_tags(text):
      clean_text = re.sub(r'@\w+', '', text)
      return clean_text
  #REMOVING CHATWORDS
def replace_chat_words(text):
      chat_words = {
          "BRB": "Be right back",
          "BTW": "By the way",
          "OMG": "Oh my God/goodness",
          "TTYL": "Talk to you later",
          "OMW": "On my way",
          "SMH": "Shaking my head", "SMDH": "Shaking my darn head",
          "LOL": "Laugh out loud",
          "ROFL": "Rolling on the floor laughing",
          "LMAO": "Laughing my ass off",
          "LMFAO": "Laughing my freaking ass off",
          "TBD": "To be determined",
          "IMHO": "In my humble opinion", "IMO": "In my opinion",
          "HMU": "Hit me up",
          "IIRC": "If I remember correctly",
          "LMK": "Let me know",
          "OG": "Original gangsters (used for old friends)",
          "FTW": "For the win",
          "NVM": "Nevermind",
          "OOTD": "Outfit of the day",
          "NGL": "Not gonna lie",
          "RQ": "Real quick",
          "IYKYK": "If you know, you know",
          "ONG": "On God (I swear)",
          "YAAAS": "Yes!",
          "BRT": "Be right there",
          "SM": "So much",
          "IG": "I guess",
          "WYA": "Where you at",
          "ISTG": "I swear to God",
          "HBU": "How about you",
          "ATM": "At the moment",
          "ASAP": "As soon as possible",
          "FYI": "For your information",
          "TBH": "To be honest",
          "IDC": "I don't care",
          "IDK": "I don't know",
          "ILY": "I love you",
          "IMU": "I miss you",
          "JK": "Just kidding",
          "TMI": "Too much information",
          "GTG": "Got to go",
          "G2G": "Got to go",
          "BFF": "Best friends forever",
          "TTYT": "Talk to you tomorrow",
          "GG": "Good game",
          "GLHF": "Good luck, have fun",
          "WTF": "What the freak",
          "FML": "Freak my life",
          "IDC": "I don't care",
          "DM": "Direct message",
          "PM": "Private message",
          "BTS": "Behind the scenes",
          "STFU": "Shut the freak up",
          "GRWM": "Get ready with me",
          "FWIW": "For what it's worth",
          "YOLO": "You only live once",
          "TT": "Throwback Thursday",
          "MFW": "My face when",
          "SMTH": "Something",
          "WB": "Welcome back",
          "SYS": "See you soon",
          "CW": "Content warning",
          "NSFW": "Not safe for work",
          "TBF": "To be fair",
          "FOMO": "Fear of missing out",
          "GOAT": "Greatest of all time",
          "SUS": "Suspicious",
          "B4": "Before",
          "THX": "Thanks",
          "NP": "No problem",
          "PLS": "Please",
          "K": "Okay",
          "KK": "Okay",
          "XOXO": "Hugs and kisses"
      }
      def replace_match(match):
          return chat_words[match.group(0).upper()]

      pattern = r'\b(' + '|'.join(re.escape(word) for word in chat_words.keys()) + r')\b'
      return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

  #REMOVE PUNCTUAION
string.punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)
  #REMOVE STOPWORDS
from nltk.corpus import stopwords
def remove_stopwords(text):
      stop_words = set(stopwords.words('english'))
      words = text.split()
      filtered_words = [word for word in words if word.lower() not in stop_words]
      return ' '.join(filtered_words)
  #REMOVE WHITESPACE
def remove_whitespace(text):
      return text.strip()

  #REMOVE SPECIAL CHARECTERS
def remove_special_characters(text):
      clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
      return clean_text
  #LEMMATIZATION
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
      words = text.split()
      words = [lemmatizer.lemmatize(word) for word in words]
      return " ".join(words)
  #DATA CLEANING
def preprocess_text(text):
      text = remove_twitter_tags(text)
      text = remove_html(text)
      text = remove_url(text)
      text = remove_special_characters(text)
      text = convert_to_lowercase(text)
      text = replace_chat_words(text)
      text = remove_punctuation(text)
      text = remove_stopwords(text)
      text = lemmatize_text(text)
      text = remove_whitespace(text)

      return text
df['text'] = df['text'].apply(preprocess_text)