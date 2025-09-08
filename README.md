# Context Keeper 🧠

> AI Memory Layer for Development Teams - Never lose context again

## 🎯 What is Context Keeper?

Context Keeper captures, indexes, and makes queryable ALL your development context:
- Git commits and code changes
- Debugging sessions and error logs
- Architectural decisions and discussions
- Documentation and configuration changes
- Team conversations and decisions

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Node.js 18+ (for frontend)
- Git

### Installation

1. **Clone and setup:**
```bash
git clone https://github.com/yourusername/context-keeper.git
cd context-keeper
```

2. **Start infrastructure:**
```bash
cd docker
docker-compose up -d
cd ..
```

3. **Setup Python environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Initialize databases:**
```bash
python -m app.db.init_db
```

5. **Run the API server:**
```bash
python -m app.main
```

6. **Start collecting context:**
```bash
# In another terminal
cd collectors/git
python git_collector.py --repo /path/to/your/repo --history 100
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/api/docs)
- [Architecture Guide](docs/architecture.md)
- [Collector Setup](docs/collectors.md)
- [Query Examples](docs/queries.md)

## 🏗️ Tech Stack

- **FastAPI** - High-performance API server
- **Qdrant** - Vector database for semantic search
- **Neo4j** - Graph database for relationships
- **Redis** - Caching and job queue
- **LangChain** - LLM orchestration
- **Sentence Transformers** - Embeddings

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

Built with ❤️ by developers, for developers
