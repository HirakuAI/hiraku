# HIRAKU
AI LEARNING PLATFORM FOR UON STUDENTS;

----

Note:

Here is the central repo for Hiraku project. if you are the team that choose to continue this project, i wish you the best :) 

before you starting development, please read our Handover documentation first, which should provided by your sponser (Jade Ong);

here is our overall tech-stack, before develop hiraku, you should go learn the tech below first:

but we updated the tech stack, new one is:

Frontend:

  - SvelteKit (Svelte framework for building web applications)
  - TypeScript (typed superset of JavaScript)
  - Vite (frontend build tool)
  - Tailwind CSS (utility-first CSS framework)
  - PostCSS (CSS processing)
  - Cypress (end-to-end testing)
  - Vitest (unit testing)
  - i18next (internationalization)
  - Tiptap (rich text editor)
  - CodeMirror (code editor integration)
  - PWA support (Progressive Web App features)


Backend:
  - Python 3.11+
  - FastAPI (web framework)
  - Uvicorn (ASGI server)
  - Pydantic (data validation)
  - SQLAlchemy, Alembic, Peewee (ORMs and migrations)
  - PostgreSQL, MySQL, MongoDB, Redis (database support)
  - Socket.IO (real-time communication)
  - Auth: python-jose, passlib, argon2-cffi, bcrypt
  - Task scheduling: APScheduler
  - Cloud/Storage: boto3 (AWS), PyMySQL, psycopg2-binary
  - Vector DBs: pgvector, chromadb, pymilvus, qdrant-client, opensearch-py, elasticsearch, pinecone
  AI/ML: openai, anthropic, google-genai, transformers, sentence-transformers, accelerate, colbert-ai, tiktoken, langchain, etc.
  Document processing: pypdf, docx2txt, python-pptx, unstructured, nltk, pandas, openpyxl, etc.
  Image/Audio: pillow, opencv-python-headless, rapidocr-onnxruntime, soundfile, onnxruntime, faster-whisper

----

here is how you can run the projects

first, you need to have a laptop that have at least 16GB RAM, 30 GB Storage, if you wanna run the model offline with Ollama (which you have too 
cuz you **wont** have enough budget to go cloude), you need to have a Nvidia RTX20 Series+ GPU with **at lease** 6GB VRAM , or Apple Sicilion M1
above chips.

we suggest you develop on the Linux/MacOS system, if you really need to use windows, please running it in WSL.

then, open terminal, `cd` do project root dir, running `npm install` to install all the essential npm front end package

next, open another terminal tab, `cd` to the `backend/` folder, running `pip install requirement.txt`, then base on your system, running the specific
starting script.

all done, Good luck for your development.

> _you either die a hero or live long enough to see yourself become the villain_


