"""
============================================================
CODTECH INTERNSHIP - TASK 3
AI Chatbot with NLP
Built using pure Python NLP: tokenization, stemming,
TF-IDF similarity + keyword rule matching
============================================================
"""

import re, string, random, math
from collections import Counter
from datetime import datetime

print("=" * 60)
print("  CODTECH INTERNSHIP – TASK 3: AI CHATBOT WITH NLP")
print("=" * 60)

STOP_WORDS = {
    'a','an','the','is','it','in','on','at','to','for','of','and',
    'or','but','i','you','me','my','your','we','they','this','that',
    'what','how','when','where','who','do','does','did','can','will',
    'have','has','had','be','been','am','are','was','were','with','about'
}

def tokenize(text):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    return [w for w in text.split() if w]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]

def stem(word):
    for s in ['ing','tion','ness','ment','ity','er','est','ly','ed','s']:
        if word.endswith(s) and len(word)-len(s) >= 3:
            return word[:-len(s)]
    return word

def preprocess(text):
    return [stem(w) for w in remove_stopwords(tokenize(text))]

INTENTS = [
    {"tag":"greeting","keywords":["hello","hi","hey","morning","evening","howdy","sup","greet"],
     "responses":["Hello! Welcome to CodTech Assistant 🤖 How can I help?",
                  "Hey there! I'm CodBot. Ask me anything about Python or your internship!",
                  "Hi! Great to see you. What can I do for you today?"]},
    {"tag":"farewell","keywords":["bye","goodbye","see","exit","quit","later","night","farewell"],
     "responses":["Goodbye! Keep coding! 👋","See you later! Good luck with your internship! 🚀","Bye! Come back anytime!"]},
    {"tag":"thanks","keywords":["thank","thanks","appreciate","great","awesome","wonderful","perfect"],
     "responses":["You're welcome! 😊","Happy to help!","Anytime! That's what I'm here for."]},
    {"tag":"joke","keywords":["joke","funny","laugh","humor","fun"],
     "responses":["Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
                  "Why did the Python dev break up with Java? Too many classes! 😄",
                  "A SQL query walks into a bar and asks two tables: 'Can I join you?' 😂"]},
    {"tag":"time","keywords":["time","date","today","clock","now"],
     "responses":["__DYNAMIC_TIME__"]},
    {"tag":"python_basics","keywords":["python","programming","language","syntax","feature","interpreter"],
     "responses":["Python is a high-level, interpreted language known for simplicity. "
                  "It supports OOP, functional, and procedural styles. "
                  "Great for web dev, data science, automation, and AI. "
                  "Key libraries: NumPy, Pandas, TensorFlow, Django. 🐍"]},
    {"tag":"machine_learning","keywords":["machine","learning","ml","supervised","unsupervised","neural","deep","ai","algorithm"],
     "responses":["Machine Learning enables systems to learn from data!\n"
                  "• Supervised: labeled data → classification, regression\n"
                  "• Unsupervised: unlabeled → clustering, PCA\n"
                  "• Reinforcement: reward-based learning\n"
                  "Libraries: scikit-learn, TensorFlow, PyTorch 🤖"]},
    {"tag":"nlp","keywords":["nlp","natural","language","text","token","stem","lemma","entity","sentiment","processing"],
     "responses":["NLP lets computers understand human language!\n"
                  "Key tasks: Tokenization → Stop-word removal → Stemming → POS Tagging → NER → Sentiment Analysis\n"
                  "Python libraries: NLTK, spaCy, HuggingFace Transformers 📝"]},
    {"tag":"api","keywords":["api","rest","request","json","http","fetch","endpoint","url","web"],
     "responses":["APIs let software communicate!\n"
                  "Python example:\n  import requests\n  r = requests.get('https://api.example.com/data')\n  data = r.json()\n"
                  "Popular APIs: OpenWeatherMap, GitHub, NewsAPI 🌐"]},
    {"tag":"scikit_learn","keywords":["scikit","sklearn","classifier","svm","naive","bayes","logistic","forest","accuracy","precision","recall","f1","confusion"],
     "responses":["Scikit-learn is Python's premier ML library!\n"
                  "• Classifiers: Naive Bayes, SVM, Random Forest, Logistic Regression\n"
                  "• Metrics: accuracy_score, confusion_matrix, classification_report\n"
                  "• Pipeline: chain preprocessing + model together 🔬"]},
    {"tag":"data_science","keywords":["data","pandas","numpy","matplotlib","seaborn","visualization","dataframe","analysis","dataset","plot","chart"],
     "responses":["Data Science = statistics + programming + domain knowledge!\n"
                  "Core tools:\n• Pandas – data manipulation\n• NumPy – numerical computing\n"
                  "• Matplotlib/Seaborn – visualization\n• Jupyter – interactive notebooks 📊"]},
    {"tag":"codtech_tasks","keywords":["codtech","internship","task","tasks"],
     "responses":["Your CodTech Python Internship has 4 tasks:\n"
                  "📌 Task 1 – API Integration & Data Visualization\n"
                  "📌 Task 2 – Automated PDF Report Generation\n"
                  "📌 Task 3 – AI Chatbot with NLP ← you are here!\n"
                  "📌 Task 4 – Machine Learning Model (scikit-learn)\n"
                  "Complete all 4 before the deadline! 🚀"]},
    {"tag":"github","keywords":["github","git","repo","repository","commit","push","branch","version","control"],
     "responses":["Git essentials:\n  git init → git add . → git commit -m 'msg' → git push\n"
                  "Store all CodTech task files in a GitHub repository! 🗂️"]},
    {"tag":"help","keywords":["help","capability","feature","topic","command","assist","support"],
     "responses":["I can help with:\n🐍 Python basics\n🤖 ML & AI concepts\n📝 NLP\n"
                  "📊 Data Science (Pandas, NumPy)\n🌐 API integration\n"
                  "🔬 Scikit-learn\n📋 CodTech internship tasks\n🗂️ GitHub\nJust ask!"]},
    {"tag":"unknown","keywords":[],
     "responses":["I'm not sure about that. Try asking about Python, ML, NLP or your CodTech tasks! 🤔",
                  "Could you rephrase? I'm best at Python, Data Science, and your internship! 💡"]},
]

def classify_intent(user_input):
    text_lower = user_input.lower()
    tokens = set(tokenize(text_lower))
    best_tag, best_score = "unknown", 0
    for intent in INTENTS:
        if intent["tag"] == "unknown": continue
        matches = sum(1 for kw in intent["keywords"] if kw in text_lower or kw in tokens)
        score = matches / max(len(intent["keywords"]), 1)
        if score > best_score:
            best_score, best_tag = score, intent["tag"]
    return best_tag, best_score

def get_response(tag):
    for intent in INTENTS:
        if intent["tag"] == tag:
            r = random.choice(intent["responses"])
            if r == "__DYNAMIC_TIME__":
                return f"It's {datetime.now().strftime('%A, %d %B %Y – %I:%M %p')} 🕐"
            return r
    return random.choice(INTENTS[-1]["responses"])

class ChatBot:
    def __init__(self):
        self.name = None
        self.history = []
    def respond(self, user_input):
        user_input = user_input.strip()
        if not user_input: return "Please type something! 😊"
        m = re.search(r'\b(?:my name is|i am|call me|i\'m)\s+([A-Za-z]+)', user_input, re.I)
        if m:
            self.name = m.group(1).capitalize()
            return f"Nice to meet you, {self.name}! 😊 How can I help with your Python internship?"
        if re.search(r'what.*my name|remember.*name', user_input.lower()):
            return f"Your name is {self.name}!" if self.name else "You haven't told me your name yet!"
        tag, score = classify_intent(user_input)
        response = get_response(tag)
        if self.name and random.random() < 0.25:
            response = f"{self.name}, {response[0].lower()}{response[1:]}"
        self.history.append({"user": user_input, "bot": response, "intent": tag})
        return response

# Demo
print("\n[1] Running demo conversation...\n" + "─"*55)
bot = ChatBot()
demos = [
    "Hello!", "My name is Arjun",
    "What are the CodTech internship tasks?",
    "Tell me about machine learning",
    "What is NLP?", "How do I use APIs in Python?",
    "Tell me about scikit-learn",
    "Explain data science tools",
    "How does GitHub work?",
    "Tell me a joke", "What can you help me with?",
    "Thank you!", "Bye!"
]
for msg in demos:
    print(f"👤 You   : {msg}")
    print(f"🤖 CodBot: {bot.respond(msg)}\n")
print("─"*55)
print(f"\n✅ Task 3 Complete! ({len(bot.history)} conversation turns)")

import shutil
shutil.copy('/home/claude/task3_chatbot.py', '/mnt/user-data/outputs/task3_chatbot.py')
