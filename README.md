# CivitAI Model Downloader

A robust, multithreaded command-line tool for bulk-downloading models, images, and metadata from [CivitAI](https://civitai.com) — by username or by direct model ID.

Built on top of the original script by [Confuzu](https://github.com/Confuzu), with various added features and support for [galCME](https://github.com/LeonWGal/galCME).

[Русская версия](README.ru.md)

## Features

- **Download by username or model ID** — grab everything a creator has uploaded, or fetch specific models directly with `--model_id`.
- **Resumable downloads** — interrupted transfers resume via HTTP `Range` requests instead of restarting from zero.
- **Integrity verification** — every model file is checked against the size/SHA256 hash reported by the API; corrupted or truncated files are automatically re-downloaded. A background scanner (`--deep_check`) can re-verify your entire library on every run.
- **Slow-download auto-restart** — a transfer stuck below 100 KB/s for 30 seconds is killed and retried automatically.
- **Generation metadata for example images** — prompt, seed, sampler, and model info are saved next to each example image as `{id}_meta.txt` (or `{id}_no_meta.txt` when unavailable).
- **Duplicate detection & cleanup** — removes duplicate files and duplicate example images (matched by CivitAI image ID), preferring the copy with the more descriptive filename.
- **`.skip` marker files** — drop a `.skip` file into any user/model/version folder to permanently exclude it from future runs.
- **Flexible filtering** — by content type (`Lora`, `Checkpoints`, `Embeddings`, `Training_Data`, `Other`, `All`), by exclusion, and by base model (fuzzy, comma-separated, e.g. `--base_models Illustrious,Pony`).
- **Path-traversal hardened** — usernames, filenames, and folder names coming from the API are strictly sanitized and every constructed path is verified to stay inside the output directory before any file is written.
- **Secure token handling** — the API token is never echoed to the terminal: pass `--token`, set `CIVITAI_API_TOKEN`, or enter it via a hidden prompt.
- **Long-path safe** — automatically truncates and migrates overly long folder/file names for Windows compatibility.
- **Live progress UI** — a per-user overall progress bar plus a live "closest to completion" file transfer bar, followed by a colored summary.

## Requirements

```
pip install requests tqdm
```

Python 3.8+ is recommended (uses the walrus operator).

## Usage

```bash
python civitAI_Model_downloader.py <username> [<username2> ...] [options]
python civitAI_Model_downloader.py --model_id 12345,67890 [options]
```

### Authentication

The script needs a CivitAI API token (create one at civitai.com → Account Settings → API Keys). Priority order:

1. `--token YOUR_TOKEN`
2. `CIVITAI_API_TOKEN` environment variable
3. Hidden interactive prompt (input is not echoed)

### Options

| Flag | Default | Description |
|---|---|---|
| `usernames` | — | One or more CivitAI usernames (positional, omit if using `--model_id`) |
| `--model_id`, `--model_ids` | — | Model ID or comma-separated IDs to download directly, bypassing username search |
| `--download_type` | `All` | One of `Lora`, `Checkpoints`, `Embeddings`, `Training_Data`, `Other`, `All` |
| `--exclude_type` | — | Download everything except this type (mutually exclusive with `--download_type`) |
| `--base_models` | — | Comma-separated base model filter, matched case-insensitively as a substring (e.g. `Illustrious,Pony,SDXL`) |
| `--max_threads` | `5` | Concurrent download threads |
| `--max_tries` | `3` | Max retries per file |
| `--retry_delay` | `10` | Seconds between retries (exponential backoff applied) |
| `--deep_check` | off | Verify SHA256 hashes of existing files at startup (slow, thorough) |
| `--token` | — | API token (prefer the env var instead) |

### Examples

```bash
# Download all LoRAs from a user
python civitAI_Model_downloader.py someartist --download_type Lora

# Download everything except checkpoints, from two users
python civitAI_Model_downloader.py userA userB --exclude_type Checkpoints

# Download two specific models by ID
python civitAI_Model_downloader.py --model_id 12345,67890

# Only Illustrious/Pony-based models, with more threads
python civitAI_Model_downloader.py someartist --base_models Illustrious,Pony --max_threads 8

# Full integrity re-check of an existing library
python civitAI_Model_downloader.py someartist --deep_check
```

## Output layout

```
model_downloads/
└── <username>/
    └── <category>/            # Lora, Checkpoints, Embeddings, Training_Data, Other
        └── <base_model>/      # e.g. Illustrious, Pony, SDXL 1.0 (if reported)
            └── <model name>/
                └── <version name>/
                    ├── <file>.safetensors
                    ├── <file>.civitai.info      # full API metadata for this version
                    ├── <file>.json              # description + trigger words
                    ├── triggerWords.txt
                    ├── <name>.preview.jpg
                    └── examples/
                        ├── <name>_<id>.jpeg
                        └── <id>_meta.txt         # or <id>_no_meta.txt
```

Logs, per-username summaries, and failed-download reports are written to `logs/`.

## Skipping items

Create an empty file named `.skip` inside any `username/`, `.../<model>/`, or `.../<model>/<version>/` folder to have the script permanently skip that user, model, or version on future runs.

## Security notes

- The API token is sent only via the `Authorization` header, never appended to URLs.
- All filenames and usernames originating from API responses are sanitized and path-joined with a containment check (`safe_path_join`) before touching the filesystem.
- Pagination only follows `nextPage` URLs whose host is `civitai.com` / `www.civitai.com`, preventing token exfiltration via a malicious or spoofed API response.

## Credits

Originally based on [Confuzu's CivitAI Model Downloader](https://github.com/Confuzu). This fork has diverged significantly with its own set of reliability, integrity, and UX features.

## License

Use at your own risk. Respect CivitAI's [Terms of Service](https://civitai.com/content/tos) and the license terms of each model you download.
