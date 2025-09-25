# Story Summarizer

A Python script to recursively summarize large text files (stories, transcripts, documents).  
It works with **LM Studio by default**, but also supports other backends like **Ollama** or **llama.cpp** that expose the OpenAI-style `/v1/chat/completions` API.

---

##  Features
- Handles very large files by **chunking and recursive summarization**.
- Adjustable summary detail (`-s 1–5`).
- Works out-of-the-box with **LM Studio**.
- Switchable backend: **Ollama**, **llama.cpp**, **vLLM**, or even OpenAI.
- Optionally keeps **intermediate chunk summaries** (`-k`).

---

##  Installation

Clone the repository and install dependencies:


git clone https://github.com/<your-username>/story-summarizer.git
cd story-summarizer
pip install -r requirements.txt

## Usage
LM Studio (default)
python summarize.py my_story.txt

Ollama
python summarize.py my_story.txt \
  --base-url http://127.0.0.1:11434/v1 \
  --model mistral:7b

--
Options

-s LEVEL → Summary detail level (1–5, default: 5 = full detail).

-k → Keep intermediate chunk summaries. Useful if you want to see how the individual chunks of the text were summarized before being recursively summarized. (chunk_summaries_<file>.txt).

-c TOKENS → Context window size of the model (default: 32000). THIS IS IMPORTANT. It will set the size of the chunks and the recursion levels automatically to optimize your context size.

--base-url → Backend endpoint (LM Studio, Ollama, llama.cpp, etc.).

--model → Model name (e.g. local-model, mistral:7b). Works well with Qwen3 Next 20b. Avoid thinking models if possible. 

--
 Output

For an input file my_story.txt, the script creates:

Full_Summary_my_story.txt → the full master summary.

Summary_my_story.txt → compressed version (if -s < 5).

chunk_summaries_my_story.txt → optional, intermediate summaries (if -k).



## Notes & Limitations

Token estimation is heuristic (~4 characters = 1 token).

Quality depends heavily on the chosen model’s capabilities.

If the backend is unreachable, the script will skip writing empty files and warn instead.

## License

MIT License – feel free to use, modify, and share.