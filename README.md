# ✈️ AI-Powered Airline Policy Assistant

![Multi-Airline RAG Policy Assistant Interface](streamlit_interface1.png)

> **Ask any airline policy question and get instant, accurate answers powered by AI**

Stop digging through confusing airline websites. Just ask naturally: *"Can I bring my guitar on board?"* or *"What's the pet policy for international flights?"*

## 💡 Why This Matters

**Before**: 😤 Spend 30+ minutes navigating multiple airline websites  
**After**: ⚡ Get precise answers in seconds with AI-powered search

🎯 **Real-time policy data** from major airlines  
🤖 **Natural language** - ask questions like you're talking to a travel agent  
🔍 **Smart search** - finds relevant info even with vague questions

## ⚡ Get Started in 2 Minutes

```bash
# 1. Clone & Start
git clone <repo-url> && cd multi-airline-rag-system
docker-compose up -d

# 2. Load Data (one-time setup)
docker-compose run scraper python scraper_only.py

# 3. Start Asking Questions!
# Open: http://localhost:8501
```

## 🚀 What You Get

| Feature | Benefit |
|---------|---------|
| 🎯 **Smart Q&A** | Ask complex questions, get precise answers |
| ✈️ **Multi-Airlines** | Turkish Airlines + Pegasus (more coming) |
| 🔄 **Always Updated** | Policies sync automatically from airline websites |
| 🌍 **Bilingual** | Works in English & Turkish |
| 📱 **Easy Interface** | Clean, intuitive web dashboard |

## 💻 Quick Examples

**Try these questions:**
- *"What items are banned in carry-on luggage?"*
- *"How much does excess baggage cost?"*
- *"Can I travel with my cat in the cabin?"*
- *"Compare baggage policies between airlines"*

**API Access:**
```bash
curl "http://localhost:8000/search?q=pet+travel"
curl -X POST "http://localhost:8000/chat/openai?question=baggage+fees"
```

## 🛠️ Under the Hood

**Tech**: FastAPI + PostgreSQL + OpenAI + Docker  
**How it works**: Web scraping → AI embeddings → Semantic search → Natural answers

```
Your Question → AI Search → Airline Policies → Smart Answer
```

## 🔧 Quick Fixes

**No data?** Run: `docker-compose run scraper python scraper_only.py`  
**Still loading?** Wait 2-3 minutes for full startup  
**Check status**: `curl http://localhost:8000/health`

---

## 🌟 Ready to Try?

1. **Clone** the repo
2. **Run** `docker-compose up -d`  
3. **Load** data with the scraper
4. **Ask** your first question!

**🔗 Access Points:**
- 🖥️ **Web App**: http://localhost:8501
- 📚 **API Docs**: http://localhost:8000/docs

---

**Built with ❤️ for effortless travel**