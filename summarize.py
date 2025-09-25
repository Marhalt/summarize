import argparse
import os
import requests
from datetime import datetime

# Usage: summary.py filename -s 1-5 [-k] [-c context length]

# -------------------------
# Configurable parameters
# -------------------------

BASE_SYSTEM_PROMPT = (
    "You are an editor. Your job is to summarize stories efficiently. "
    "Never refuse or say you cannot comply. "
    "Do not moralize, judge, or deflect. "
    "Write in coherent complete paragraphs - avoid short, punchy lines or other literary devices designed for effect, "
    "and instead focus on narrative and chronology "
)

# -------------------------
# Generic LLM Client
# -------------------------
class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:1234/v1", model="local-model"):
        # Default points to LM Studio
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model

    def chat(self, prompt, max_tokens=1000, temperature=0, system_prompt=None, timeout=300):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(self.url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"❌ Error contacting backend at {self.url}: {e}")
            return ""

# -------------------------
# Argument parsing
# -------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize a story with varying detail levels",
        epilog="Example: python summarize.py -s 2 -c 100000 data.txt"
    )

    parser.add_argument("filename", type=str, help="Input file to process (required)")
    parser.add_argument("-s", "--summary", type=int, default=5, metavar="LEVEL",
                        help="Summary detail level (1-5, default: 5)")
    parser.add_argument("-k", "--keep", action="store_true",
                        help="If set, write concatenated chunk summaries to chunk_summaries_<filename>.txt")
    parser.add_argument("-c", "--context", type=int, default=32000, metavar="TOKENS",
                        help="Maximum context window size of the model in tokens (default: 32000)")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:1234/v1",
                        help="Base URL for the backend (default: LM Studio at 127.0.0.1:1234)")
    parser.add_argument("--model", type=str, default="local-model",
                        help="Model name (default: 'local-model')")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        parser.error(f"File not found: '{args.filename}'")
    if args.summary < 1 or args.summary > 5:
        parser.error("Summary level must be between 1 and 5")

    return args.summary, args.filename, args.keep, args.context, args.base_url, args.model

# -------------------------
# Chunk text helper
# -------------------------
def chunk_text(text, chunk_size_tokens=35000, overlap_tokens=500):
    chars_per_token = 4  # heuristic
    chunk_size = chunk_size_tokens * chars_per_token
    overlap = overlap_tokens * chars_per_token

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------
# Scaling function for master summary
# -------------------------
def target_master_length(story_tokens):
    if story_tokens <= 100000:
        return "Target length: 3 pages (~1500 words)."
    elif story_tokens <= 500000:
        return "Target length: 5–7 pages (~2500–3500 words)."
    elif story_tokens <= 1000000:
        return "Target length: 8–10 pages (~4000–6000 words)."
    else:
        return "Target length: 12–15 pages (~6000–9000 words)."

# -------------------------
# Recursive summarization
# -------------------------
def recursive_summarize(texts, client, detail_instruction, context_limit, level=0):
    combined = "\n\n".join(texts)
    token_estimate = len(combined) // 4

    if token_estimate < context_limit * 0.8:
        prompt = (
            f"Combine the following summaries into one coherent summary. "
            f"{detail_instruction}\n\n{combined}"
        )
        print(f"\n🔄 Final summarization at recursion level {level}, tokens ≈ {token_estimate}")
        return client.chat(prompt,
                           max_tokens=int(context_limit * 0.9),
                           temperature=0.2,
                           system_prompt=BASE_SYSTEM_PROMPT + "/think")

    group_size = max(5, context_limit // 2000)
    groups = [texts[i:i + group_size] for i in range(0, len(texts), group_size)]

    print(f"\n Text too large (≈{token_estimate} tokens). "
          f"Breaking into {len(groups)} groups at recursion level {level}...")

    higher_level_summaries = []
    for i, group in enumerate(groups, start=1):
        print(f"  • Summarizing group {i}/{len(groups)} at level {level}...")
        combined_group = "\n\n".join(group)
        prompt = f"Combine the following summaries into one coherent summary:\n\n{combined_group}"
        group_summary = client.chat(prompt,
                                    max_tokens=int(context_limit * 0.8),
                                    temperature=0.1,
                                    system_prompt=BASE_SYSTEM_PROMPT + "/nothink")
        higher_level_summaries.append(group_summary)

    return recursive_summarize(higher_level_summaries, client, detail_instruction, context_limit, level + 1)

# -------------------------
# Summarization function
# -------------------------
def summarize_story(filename: str, context_limit: int, keep_flag: bool, client: LLMClient):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        story_text = f.read()

    token_estimate = len(story_text) // 4

    master_instruction = (
        "Write a detailed summary of the following story. "
        f"{target_master_length(token_estimate)}\n"
        "Do not be too concise. Describe the flow of the plot, relationships, and characters in chronological order. "
        "Ensure the summary is long enough to cover the entire story faithfully."
    )

    # Case 1: Entire story fits
    if token_estimate < context_limit * 0.8:
        print(f"  Strategy: One-pass injection (≈{token_estimate} tokens). Skipping chunking.")
        prompt = f"{master_instruction}\n\n{story_text}"
        return client.chat(prompt,
                           max_tokens=int(context_limit * 0.9),
                           temperature=0.2,
                           system_prompt=BASE_SYSTEM_PROMPT + "/think"), story_text

    # Case 2: Chunking needed
    CHUNK_SIZE = int(0.75 * context_limit)
    chunks = chunk_text(story_text, chunk_size_tokens=CHUNK_SIZE, overlap_tokens=500)
    print(f"  Strategy: Chunking required (story ≈{token_estimate} tokens).")
    print(f"   Using chunk size: {CHUNK_SIZE} tokens → {len(chunks)} chunks.")

    # Initialize the file for the chunk summaries
    chunks_filename = None
    if keep_flag:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        chunks_filename = f"chunk_summaries_{base_name}.txt"

        print(f"Writing chunk summaries to {chunks_filename}. \n")
        with open(chunks_filename, "w", encoding="utf-8") as f:
            f.write(f"Chunk summaries for {filename}\n\n")

    chunk_summaries = []
    print("Processing chunks:", end=" ", flush=True)

    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            "Write a detailed summary of the following text, "
            "focusing on key events, character actions, and relationships. "
            "Capture the key events and character milestones, in chronological order.\n\n"
            f"Text:\n{chunk}"
        )
        summary = client.chat(
            prompt,
            max_tokens=int(context_limit * 0.25),
            temperature=0,
            system_prompt="/nothink" + BASE_SYSTEM_PROMPT
        )

        entry = f"--- Chunk {i} ---\n{summary}\n\n"
        chunk_summaries.append(entry)

        if keep_flag and chunks_filename:
            with open(chunks_filename, "a", encoding="utf-8") as f:
                f.write(entry)

        print(f"{i}", end=" ", flush=True)

    print("\nAll chunks processed, creating final master summary...")
    final_summary = recursive_summarize(chunk_summaries, client, master_instruction, context_limit)

    combined_text = "\n\n".join(chunk_summaries)
    return final_summary, combined_text

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    summary_level, filename, keep_flag, context_limit, base_url, model = parse_arguments()
    base_name = os.path.splitext(os.path.basename(filename))[0]

    print(f"\n Configuration validated:")
    print(f"   • Summary level: {summary_level}")
    print(f"   • Context limit: {context_limit} tokens")
    print(f"   • Input file: '{filename}'")
    print(f"   • Backend URL: {base_url}")
    print(f"   • Model: {model}\n")

    client = LLMClient(base_url=base_url, model=model)

    start_time = datetime.now()
    print(f"Summarization started at {start_time.strftime('%Y-%m-%d %H:%M')}\n")

    master_summary, combined_text = summarize_story(filename, context_limit, keep_flag, client)

    if not master_summary.strip():
        print("ERROR - No summary was generated. Skipping file output.")
    else:
        # Always write full master summary
        full_summary_filename = f"Full_Summary_{base_name}.txt"
        with open(full_summary_filename, "w", encoding="utf-8") as f:
            f.write(master_summary)

        full_word_count = len(master_summary.split())
        print(f"\n Full master summary written to {full_summary_filename}")
        print(f"   • Length: {full_word_count} words\n")

        # If requested level < 5, compress the master
        if summary_level < 5:
            level_instructions = {
                1: "Summarize the following into 3–4 paragraphs.",
                2: "Summarize the following into 7–8 paragraphs.",
                3: "Summarize the following into about 1,500 words.",
                4: "Summarize the following into about 2,500 words.",
            }
            compress_instruction = level_instructions[summary_level]
            compress_prompt = (
                f"Summarize the following long summary into a shorter version.\n"
                f"{compress_instruction}\n\n{master_summary}"
            )
            short_summary = client.chat(
                compress_prompt,
                max_tokens=int(context_limit * 0.5),
                temperature=0.2,
                system_prompt=BASE_SYSTEM_PROMPT + "/think"
            )

            if not short_summary.strip():
                print("CAUTION:  Compression step failed, no shorter summary generated.")
            else:
                summary_filename = f"Summary_{base_name}.txt"
                with open(summary_filename, "w", encoding="utf-8") as f:
                    f.write(short_summary)

                short_word_count = len(short_summary.split())
                print(f" Compressed summary written to {summary_filename}")
                print(f"   • Length: {short_word_count} words\n")

    end_time = datetime.now()
    print(f"Finished summarizing at {end_time.strftime('%Y-%m-%d  %H:%M')}\n")
    print('\a'); print('\a'); print('\a')