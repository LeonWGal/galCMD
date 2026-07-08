import re
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import urllib.parse
import os
import sys
import shutil
import threading
import getpass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
import time
import argparse
import hashlib

# ============================================================
# Constants
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = os.getenv("MODEL_EXPLORER_ROOT", "E:/model_downloads" if os.path.exists("E:/model_downloads") else "model_downloads")
TEMP_DIR = os.path.join(SCRIPT_DIR, ".downloading")
MAX_PATH_LENGTH = 260  # Windows max path
MAX_COMPONENT_LENGTH = 80  # Max length for a single path component (folder/file name)
CHUNK_SIZE = 1048576  # 1MB chunks for faster downloads
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All']
BASE_URL = "https://civitai.com/api/v1/models"
ALLOWED_API_HOSTS = {'civitai.com', 'www.civitai.com'}
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
SKIP_MARKER = ".skip"  # Marker file — if present in a folder, skip downloading

# Slow download detection
SLOW_SPEED_THRESHOLD = 100 * 1024   # 100 KB/s
SLOW_SPEED_TIMEOUT = 30             # seconds below threshold to trigger restart

# Console formatting helpers
class Style:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    WHITE = '\033[97m'
    MAGENTA = '\033[35m'

# Thread-safe locks
_print_lock = threading.Lock()
_failed_file_lock = threading.Lock()

# Shared download progress tracking — threads update this dict,
# main thread reads it to display the "closest to completion" bar.
_active_downloads = {}  # thread_id -> {'name': str, 'downloaded': int, 'total': int}
_active_downloads_lock = threading.Lock()

def _register_download(name, total):
    tid = threading.current_thread().ident
    with _active_downloads_lock:
        _active_downloads[tid] = {'name': name, 'downloaded': 0, 'total': total}

def _update_download_progress(downloaded):
    tid = threading.current_thread().ident
    with _active_downloads_lock:
        if tid in _active_downloads:
            _active_downloads[tid]['downloaded'] = downloaded

def _unregister_download():
    tid = threading.current_thread().ident
    with _active_downloads_lock:
        _active_downloads.pop(tid, None)

def tqdm_print(msg):
    """Thread-safe print that works with tqdm progress bars."""
    with _print_lock:
        tqdm.write(msg)
        sys.stdout.flush()

# ── Box Drawing UI ──
def ui_banner():
    """Print the main application banner."""
    print(f"{Style.CYAN}{Style.BOLD}")
    print(f"  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║          🎨  CivitAI Model Downloader               ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"{Style.RESET}")

def ui_config(users, dtype, exclude, bmodels):
    """Print configuration summary line."""
    parts = [f"{Style.WHITE}{len(users)} users{Style.RESET}"]
    if dtype:
        parts.append(f"{Style.CYAN}{dtype}{Style.RESET}")
    if exclude:
        parts.append(f"{Style.DIM}−{exclude}{Style.RESET}")
    if bmodels:
        parts.append(f"{Style.MAGENTA}{', '.join(bmodels)}{Style.RESET}")
    print(f"  {' │ '.join(parts)}")
    print()

def ui_phase(number, total, name, status=None, detail=None):
    """Print a phase indicator line."""
    tag = f"{Style.DIM}[{number}/{total}]{Style.RESET}"
    if status == 'done':
        icon = f"{Style.GREEN}✓{Style.RESET}"
    elif status == 'fail':
        icon = f"{Style.RED}✗{Style.RESET}"
    elif status == 'warn':
        icon = f"{Style.YELLOW}⚠{Style.RESET}"
    elif status == 'active':
        icon = f"{Style.CYAN}▶{Style.RESET}"
    else:
        icon = f"{Style.DIM}○{Style.RESET}"
    line = f"  {tag} {icon} {Style.BOLD}{name}{Style.RESET}"
    if detail:
        line += f"  {Style.DIM}{detail}{Style.RESET}"
    print(line)

def ui_user_box(username, lines):
    """Print a compact per-user result box."""
    w = 56
    header = f"─ {username} "
    header += "─" * max(0, w - len(username) - 3)
    print(f"  {Style.DIM}┌{header}┐{Style.RESET}")
    for line in lines:
        print(f"  {Style.DIM}│{Style.RESET} {line}")
    print(f"  {Style.DIM}└{'─' * w}┘{Style.RESET}")

def ui_summary(total, downloaded, skipped, failed, elapsed):
    """Print the final summary block."""
    print()
    print(f"  {Style.BOLD}{Style.CYAN}══ Summary {'═' * 44}{Style.RESET}")
    parts = []
    parts.append(f"{Style.WHITE}Total: {total}{Style.RESET}")
    if downloaded > 0:
        parts.append(f"{Style.GREEN}Downloaded: {downloaded}{Style.RESET}")
    if skipped > 0:
        parts.append(f"{Style.DIM}Skipped: {skipped}{Style.RESET}")
    if failed > 0:
        parts.append(f"{Style.RED}Failed: {failed}{Style.RESET}")
    print(f"  {' │ '.join(parts)}")
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"  {Style.DIM}Time: {mins}m {secs}s{Style.RESET}")
    print(f"  {Style.BOLD}{Style.CYAN}{'═' * 56}{Style.RESET}")

# Legacy formatters (still used in some places)
def fmt_header(text):
    return f"{Style.BOLD}{Style.CYAN}{'─' * 60}\n  {text}\n{'─' * 60}{Style.RESET}"

def fmt_ok(text):
    return f"{Style.GREEN}✓{Style.RESET} {text}"

def fmt_warn(text):
    return f"{Style.YELLOW}⚠{Style.RESET} {text}"

def fmt_error(text):
    return f"{Style.RED}✗{Style.RESET} {text}"

def fmt_info(text):
    return f"{Style.BLUE}ℹ{Style.RESET} {text}"

def fmt_user(username):
    return f"{Style.BOLD}{Style.HEADER}[{username}]{Style.RESET}"

def fmt_dim(text):
    return f"{Style.DIM}{text}{Style.RESET}"

# Set up logging
logger_md = logging.getLogger('md')
logger_md.setLevel(logging.DEBUG)
file_handler_md = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler_md.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_md.setFormatter(formatter)
logger_md.addHandler(file_handler_md)

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(description="Download model files and images from Civitai API.")
parser.add_argument("usernames", nargs='*', type=str, help="One or more usernames you want to download from (omit if using --model_id).")
parser.add_argument("--model_id", "--model_ids", dest="model_ids", type=str, default=None, help="Model ID or comma-separated model IDs to download directly (bypasses username search).")
parser.add_argument("--retry_delay", type=int, default=10, help="Retry delay in seconds.")
parser.add_argument("--max_tries", type=int, default=3, help="Maximum number of retries.")
parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent threads. Too many produces API Failure.")
parser.add_argument("--token", type=str, default=None, help="API Token for Civitai (or set CIVITAI_API_TOKEN env var).")
parser.add_argument("--base_models", type=str, help="Filter models by base model, comma-separated, matched case-insensitively as a substring (e.g. Illustrious,Pony,SDXL).")
parser.add_argument("--deep_check", action='store_true', help="Run SHA256 hash verification on all existing files at startup (slow but thorough).")

# Mutually exclusive group for filtering options
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--download_type",
    type=str,
    choices=VALID_DOWNLOAD_TYPES,
    help="Specify the type of content to download: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All'."
)
group.add_argument(
    "--exclude_type",
    type=str,
    choices=VALID_DOWNLOAD_TYPES,
    help="Download all content except the specified type (cannot use with --download_type)."
)

args = parser.parse_args()

if args.usernames and args.model_ids:
    print(fmt_error("Choose either usernames or --model_id, not both."))
    exit(1)
if not args.usernames and not args.model_ids:
    print(fmt_error("Provide one or more usernames, or --model_id."))
    exit(1)


def get_token_securely(cli_token):
    """Retrieve API token from CLI arg, environment variable, or a hidden prompt.

    Priority: CLI arg > CIVITAI_API_TOKEN env var > getpass (no echo).
    Avoids leaving the token visible in the terminal/scrollback.
    """
    if cli_token:
        return cli_token
    env_token = os.environ.get('CIVITAI_API_TOKEN')
    if env_token:
        return env_token
    try:
        entered_token = getpass.getpass("Enter your CivitAI API token: ")
    except (KeyboardInterrupt, EOFError):
        print("\nToken input cancelled.")
        sys.exit(1)
    if not entered_token:
        print(fmt_error("Token cannot be empty"))
        sys.exit(1)
    return entered_token


args.token = get_token_securely(args.token)

def sanitize_directory_name(name):
    return name.rstrip()

# Create output & temp directories.
OUTPUT_DIR = sanitize_directory_name(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Session with retry adapter and connection pool ---
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,
    pool_maxsize=10,
)
session.mount("https://", adapter)
session.mount("http://", adapter)

def validate_token(token):
    """Validate the API token by making a test request."""
    if not token or not token.strip():
        return False, "Token is empty"
    
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        test_url = f"{BASE_URL}?limit=1"
        response = session.get(test_url, headers=headers, timeout=10)
        
        if response.status_code == 401:
            return False, "Invalid token: Unauthorized"
        elif response.status_code == 403:
            return False, "Invalid token: Forbidden"
        elif response.status_code == 200:
            return True, "Token is valid"
        else:
            return False, f"Unexpected response: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Token validation timeout - check your internet connection"
    except requests.exceptions.ConnectionError:
        return False, "Token validation failed - check your internet connection"
    except Exception as e:
        return False, f"Token validation error: {str(e)}"

# Validate token before proceeding
is_valid, message = validate_token(args.token)
if not is_valid:
    print(fmt_error(f"Token validation failed: {message}"))
    exit(1)
print(fmt_ok(f"Token validated successfully"))

# Determine filtering options.
download_type = None
exclude_type = None
if args.download_type:
    download_type = args.download_type
elif args.exclude_type:
    exclude_type = args.exclude_type
else:
    download_type = 'All'

# Initialize variables.
usernames = args.usernames
retry_delay = args.retry_delay
max_tries = args.max_tries
max_threads = args.max_threads
token = args.token


def validate_next_page_url(url):
    """Return url if its host is on the CivitAI allowlist, else None.

    The API's pagination 'nextPage' field is followed automatically with the
    Authorization header attached. Without a host check, a compromised or
    spoofed API response could redirect requests (and the bearer token) to
    an attacker-controlled host.
    """
    if not url:
        return None
    try:
        host = urllib.parse.urlparse(url).hostname
    except ValueError:
        host = None
    if host not in ALLOWED_API_HOSTS:
        logger_md.error(f"Blocked nextPage URL with disallowed host: {host}")
        return None
    return url


def normalize_filter_value(value):
    """Normalize user-provided filter text for case/whitespace-insensitive matching."""
    return " ".join(value.strip().lower().split())


def base_model_matches(base_model, filters):
    """Return True when a model version's baseModel matches any requested filter.

    Uses substring matching plus a vowel-relaxed fallback so naming variants
    (e.g. "Illustrious" vs "IllustriousXL") still match, without requiring a
    hardcoded whitelist that goes stale as CivitAI adds new architectures.
    """
    if not filters:
        return True
    if not base_model or not isinstance(base_model, str):
        return False
    normalized_base_model = normalize_filter_value(base_model)
    for filter_value in filters:
        if filter_value in normalized_base_model:
            return True
        relaxed_filter = filter_value.rstrip("aeiou")
        if len(relaxed_filter) >= 5 and relaxed_filter in normalized_base_model:
            return True
    return False


base_models = None
if args.base_models:
    base_models = [normalize_filter_value(m) for m in args.base_models.split(',') if m.strip()]

model_ids = []
if args.model_ids:
    for raw_id in args.model_ids.split(','):
        raw_id = raw_id.strip()
        if not raw_id:
            continue
        if not raw_id.isdigit():
            print(fmt_error(f"Invalid model ID: {raw_id} (must be a number)"))
            exit(1)
        model_ids.append(int(raw_id))


# ============================================================
# File name and path utilities
# ============================================================

def safe_path_join(base_dir, *parts):
    """Join paths and verify the result stays within base_dir (blocks path traversal).

    Uses realpath() to resolve symlinks and commonpath() for robust containment
    checking. Raises ValueError if the resulting path would escape base_dir.
    """
    full_path = os.path.realpath(os.path.join(base_dir, *parts))
    base_dir_real = os.path.realpath(base_dir)
    try:
        common = os.path.commonpath([base_dir_real, full_path])
        if common != base_dir_real:
            raise ValueError(f"Path traversal blocked: {full_path}")
    except ValueError:
        # commonpath raises ValueError if paths are on different drives (Windows)
        raise ValueError(f"Path traversal blocked: {full_path}")
    return full_path


def sanitize_username_for_path(username):
    """Validate/normalize a username before using it to build filesystem paths.

    The raw username (unsanitized) should still be used for the actual API
    query — this sanitized form is only for directory/file names.
    """
    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string")

    safe = re.sub(r'[^a-zA-Z0-9_\-.]', '_', username)

    if '..' in safe or '/' in safe or '\\' in safe:
        raise ValueError(f"Invalid username: path traversal detected in '{username}'")

    safe = safe.strip('_.')
    if not safe:
        raise ValueError(f"Invalid username: '{username}' is empty after sanitization")

    if safe.upper() in RESERVED_NAMES:
        raise ValueError(f"Invalid username: '{username}' is a reserved system name")

    return safe[:50]


def sanitize_filename_strict(filename):
    """Strict filename validation for names coming directly from API responses."""
    if not filename:
        return filename

    # Extract just the basename (drops any directory components outright)
    filename = os.path.basename(filename)

    if '..' in filename:
        raise ValueError(f"Path traversal detected in filename: {filename}")

    filename = re.sub(r'[<>:"|?*\x00-\x1f\x7f-\x9f]', '_', filename)

    if not filename.strip('_. '):
        raise ValueError("Filename invalid after sanitization")

    return filename.strip()


def normalize_filename(filename):
    """Normalize filename by reducing multiple underscores/spaces to single ones."""
    base_name, extension = os.path.splitext(filename)
    # Strip Windows-invalid characters (safety net)
    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
    base_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', base_name)
    base_name = re.sub(r' +', ' ', base_name)
    base_name = re.sub(r'_+', '_', base_name)
    base_name = base_name.strip(' _')
    return base_name + extension


def truncate_path_component(name, max_len=MAX_COMPONENT_LENGTH):
    """Truncate a path component to max_len characters, preserving extension and word boundaries."""
    base_name, extension = os.path.splitext(name)
    max_base = max_len - len(extension)
    if max_base <= 0:
        max_base = max_len
        extension = ''
    
    if len(base_name) <= max_base:
        return name
    
    # Truncate and find last separator (space, underscore, hyphen)
    truncated = base_name[:max_base]
    last_sep = max(truncated.rfind(' '), truncated.rfind('_'), truncated.rfind('-'))
    if last_sep > max_base // 2:  # Only if separator isn't too early
        truncated = truncated[:last_sep]
    
    return truncated.rstrip(' _-') + extension


def sanitize_name(name, folder_name=None, max_length=MAX_COMPONENT_LENGTH):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)
    
    if folder_name and base_name == folder_name:
        return normalize_filename(name)
    
    if folder_name:
        base_name = base_name.replace(folder_name, "").strip("_")
    
    # Replace invalid Windows characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)
    base_name = base_name.strip('.')
    base_name = re.sub(r'_+', '_', base_name)
    base_name = base_name.strip('_')
    
    # Windows reserved names
    reserved_names = RESERVED_NAMES
    if base_name.upper() in reserved_names:
        base_name = '_' + base_name
    
    if not base_name:
        base_name = 'unnamed'
    
    sanitized_name = base_name + extension
    result = normalize_filename(sanitized_name.strip())
    return truncate_path_component(result, max_length)


def ensure_path_length(filepath):
    """Ensure the full path does not exceed MAX_PATH_LENGTH."""
    if len(filepath) <= MAX_PATH_LENGTH:
        return filepath
    
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base_name, ext = os.path.splitext(filename)
    
    overflow = len(filepath) - MAX_PATH_LENGTH
    new_len = max(10, len(base_name) - overflow - 5)
    truncated = base_name[:new_len].rstrip(' _-')
    
    return os.path.join(directory, truncated + ext)


# ============================================================
# Long directory name migration
# ============================================================

def migrate_long_directories(root_dir):
    """Rename existing directories whose names exceed MAX_COMPONENT_LENGTH."""
    if not os.path.exists(root_dir):
        return
    
    renamed_count = 0
    all_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname in ('examples', '.downloading'):
                continue
            full_path = os.path.join(dirpath, dirname)
            all_dirs.append((dirpath, dirname, full_path))
    
    for dirpath, dirname, full_path in all_dirs:
        if len(dirname) > MAX_COMPONENT_LENGTH:
            new_name = truncate_path_component(dirname, MAX_COMPONENT_LENGTH)
            new_path = os.path.join(dirpath, new_name)
            
            if full_path == new_path:
                continue
            
            if os.path.exists(new_path):
                logger_md.warning(f"Cannot rename '{dirname}' -> '{new_name}': target already exists")
                continue
            
            try:
                os.rename(full_path, new_path)
                renamed_count += 1
                print(fmt_info(f"Renamed: {dirname[:40]}... -> {new_name}"))
            except Exception as e:
                logger_md.error(f"Error renaming directory {full_path}: {e}")
    
    if renamed_count > 0:
        print(fmt_ok(f"Migrated {renamed_count} long directory names"))
    return renamed_count


# ============================================================
# Hashing and integrity verification
# ============================================================

def get_file_sha256(filepath):
    """Calculate SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                sha256.update(chunk)
        return sha256.hexdigest().upper()
    except Exception as e:
        logger_md.error(f"Error calculating SHA256 for {filepath}: {e}")
        return None


def verify_file_integrity(filepath, expected_hash=None, expected_size=None):
    """Verify file integrity using SHA256 hash and/or file size."""
    if not os.path.exists(filepath):
        return False
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False
    
    # Quick size check first
    if expected_size and expected_size > 0:
        size_diff = abs(file_size - expected_size)
        if size_diff > expected_size * 0.01:  # 1% tolerance
            return False
    
    # Reliable SHA256 check (slower but definitive)
    if expected_hash:
        actual_hash = get_file_sha256(filepath)
        if actual_hash and actual_hash != expected_hash.upper():
            return False
    
    return True


# ============================================================
# Temp directory and safe download
# ============================================================

def cleanup_temp_dir():
    """Remove all .part files from the temp directory (leftover from interrupted downloads)."""
    if not os.path.exists(TEMP_DIR):
        return
    
    removed = 0
    for f in os.listdir(TEMP_DIR):
        if f.endswith('.part'):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
                removed += 1
            except (OSError, PermissionError):
                pass
    
    if removed > 0:
        print(fmt_info(f"Cleaned up {removed} incomplete download(s) from previous session"))


def get_temp_path(url, output_path):
    """Generate a unique temp file path based on URL and output path."""
    unique_key = f"{url}:{output_path}"
    name_hash = hashlib.md5(unique_key.encode()).hexdigest()[:16]
    _, ext = os.path.splitext(output_path)
    return os.path.join(TEMP_DIR, f"{name_hash}{ext}.part")


# ============================================================
# Preview and duplicate handling
# ============================================================

def check_preview_exists(directory, base_name):
    """Check if a preview file exists in the directory."""
    preview_patterns = [
        f"{base_name}.preview.jpg",
        f"{base_name}.preview.jpeg",
        f"{base_name}.preview.png",
        f"{base_name}.preview.webp"
    ]
    
    for pattern in preview_patterns:
        normalized_pattern = normalize_filename(pattern)
        preview_path = os.path.join(directory, normalized_pattern)
        if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
            return True
    
    return False


def find_and_remove_duplicates_in_directory(directory):
    """Find and remove duplicate files in a directory, keeping files with normalized names."""
    if not os.path.exists(directory):
        return 0
    
    files_by_normalized_name = {}
    removed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            normalized_name = normalize_filename(file)
            key = os.path.join(root, normalized_name)
            
            if key not in files_by_normalized_name:
                files_by_normalized_name[key] = []
            files_by_normalized_name[key].append(full_path)
    
    for normalized_path, file_paths in files_by_normalized_name.items():
        if len(file_paths) > 1:
            file_paths.sort(key=lambda x: (
                len(re.findall(r'_+', os.path.basename(x))),
                len(re.findall(r'  +', os.path.basename(x))),
                -os.path.getsize(x) if os.path.exists(x) else 0
            ))
            
            keep_file = file_paths[0]
            keep_size = os.path.getsize(keep_file) if os.path.exists(keep_file) else 0
            
            normalized_full_path = os.path.join(os.path.dirname(keep_file), normalize_filename(os.path.basename(keep_file)))
            if keep_file != normalized_full_path and not os.path.exists(normalized_full_path):
                try:
                    os.rename(keep_file, normalized_full_path)
                    keep_file = normalized_full_path
                except Exception as e:
                    logger_md.error(f"Error renaming {keep_file}: {e}")
            
            for duplicate in file_paths[1:]:
                if os.path.exists(duplicate):
                    try:
                        duplicate_size = os.path.getsize(duplicate)
                        if duplicate_size > keep_size * 1.1:
                            continue
                        
                        os.remove(duplicate)
                        removed_count += 1
                    except Exception as e:
                        logger_md.error(f"Error removing duplicate {duplicate}: {e}")
    
    return removed_count


# Thread-safe image cache
_image_cache = {}
_image_cache_directory = None
_image_cache_lock = threading.Lock()


def _build_image_cache(search_directory):
    """Build image cache for fast duplicate lookups."""
    global _image_cache, _image_cache_directory
    
    with _image_cache_lock:
        if _image_cache_directory == search_directory and _image_cache:
            return
        
        _image_cache = {}
        _image_cache_directory = search_directory
        
        if not os.path.exists(search_directory):
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        
        try:
            for root, dirs, files in os.walk(search_directory):
                for file in files:
                    try:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in image_extensions:
                            if '.preview.' in file.lower():
                                continue
                            
                            base = os.path.splitext(file)[0]
                            # Extract image ID — trailing digits after _ or -
                            match = re.search(r'[_-](\d+)$', base)
                            if match:
                                file_id = match.group(1)
                            else:
                                match = re.match(r'^(\d+)$', base)
                                if match:
                                    file_id = match.group(1)
                                else:
                                    continue
                            
                            if file_id not in _image_cache:
                                _image_cache[file_id] = []
                            _image_cache[file_id].append(os.path.join(root, file))
                    except (OSError, PermissionError) as e:
                        logger_md.warning(f"Error processing file {file} in cache build: {e}")
                        continue
        except (OSError, PermissionError) as e:
            logger_md.error(f"Error building image cache for {search_directory}: {e}")


def reset_image_cache():
    """Reset the image cache."""
    global _image_cache, _image_cache_directory
    with _image_cache_lock:
        _image_cache = {}
        _image_cache_directory = None


def add_to_image_cache(image_id, image_path):
    """Add a single image entry to the cache without rebuilding."""
    if not image_id:
        return
    image_id_str = str(image_id)
    with _image_cache_lock:
        if _image_cache_directory is None:
            return  # Cache not initialized yet
        if image_id_str not in _image_cache:
            _image_cache[image_id_str] = []
        _image_cache[image_id_str].append(image_path)


def check_existing_image(image_id, search_directory):
    """Check if an image with this image_id already exists in the directory."""
    if not image_id:
        return None
    
    image_id_str = str(image_id)
    
    _build_image_cache(search_directory)
    
    with _image_cache_lock:
        if image_id_str in _image_cache:
            for path in _image_cache[image_id_str]:
                if os.path.exists(path):
                    return path
    
    return None


def find_image_duplicates(directory):
    """Find duplicate images by image ID, excluding previews.
    Reuses the image cache to avoid redundant directory walks."""
    _build_image_cache(directory)
    
    with _image_cache_lock:
        return {k: list(v) for k, v in _image_cache.items() if len(v) > 1}


def has_model_name(filename):
    """Check if filename contains a model name (not just a numeric ID).
    Returns True if the filename has a descriptive name, False if it's just a number."""
    base_name = os.path.splitext(filename)[0]
    
    # Pure numeric filename — no model name
    if re.match(r'^\d+$', base_name):
        return False
    
    # Has text parts — contains a model name
    parts = base_name.split('_')
    if len(parts) >= 2:
        # e.g. "model_name_12345" — has a model name + numeric ID
        non_numeric_parts = [p for p in parts if not re.match(r'^\d+$', p)]
        if non_numeric_parts:
            return True
    
    # Single word with letters — has a model name
    if re.search(r'[a-zA-Z]', base_name):
        return True
    
    return False


# ============================================================
# Summary / categorization
# ============================================================

def read_summary_data(username):
    """Read summary data from a file in the logs subfolder."""
    summary_path = os.path.join(LOGS_DIR, f"{username}.txt")
    data = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Total - Count:' in line:
                    total_count = int(line.strip().split(':')[1].strip())
                    data['Total'] = total_count
                elif ' - Count:' in line:
                    category, count = line.strip().split(' - Count:')
                    data[category.strip()] = int(count.strip())
    except FileNotFoundError:
        pass
    return data


def categorize_item(item):
    """Categorize the item based on its type."""
    item_type = item.get("type", "").upper()
    if item_type == 'CHECKPOINT':
        return 'Checkpoints'
    elif item_type == 'TEXTUALINVERSION':
        return 'Embeddings'
    elif item_type in ['LORA', 'LYCORIS', 'DORA', 'LOCON']:
        return 'Lora'
    elif item_type == 'TRAINING_DATA':
        return 'Training_Data'
    else:
        return 'Other'


# ============================================================
# Image generation metadata (prompts, sampler, seed, etc.)
# ============================================================

def extract_image_meta(item):
    """Extract the generation-metadata dict from an API image item.

    CivitAI's images API has used two shapes over time:
        item["meta"] = {"prompt": "...", "Model": "..."}
        item["meta"] = {"id": 123, "meta": {"prompt": "...", "Model": "..."}}
    This handles both.
    """
    meta_field = item.get("meta")
    if not meta_field or not isinstance(meta_field, dict):
        return {}

    nested_meta = meta_field.get("meta")
    if nested_meta and isinstance(nested_meta, dict):
        return nested_meta

    if "prompt" in meta_field or "Model" in meta_field or "seed" in meta_field:
        return meta_field

    return {}


def fetch_image_metadata(version_id, headers):
    """Fetch generation metadata (prompt, sampler, seed, ...) for a model version's images.

    The /models endpoint's embedded images don't carry full generation params,
    so a separate call to /images is needed.

    Returns:
        dict: image_id -> extracted meta dict. Empty dict on error.
    """
    if not version_id:
        return {}

    url = f"https://civitai.com/api/v1/images?modelVersionId={version_id}&nsfw=true"
    meta_by_id = {}

    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger_md.warning(f"Could not fetch image metadata for version {version_id}: {e}")
        return {}

    for img in data.get('items', []):
        img_id = img.get('id')
        if img_id:
            meta = extract_image_meta(img)
            base_model = img.get('baseModel')
            if meta and base_model and 'Model' not in meta:
                meta = {'Model': base_model, **meta}
            meta_by_id[img_id] = meta

    return meta_by_id


def write_image_meta_file(meta, image_id, item_dir, username):
    """Write per-image metadata to '{image_id}_meta.txt', or a '_no_meta.txt'
    fallback with a link to the image page when no generation data is available."""
    if meta and not all(str(v).strip() == '' for v in meta.values()):
        filename = f"{image_id}_meta.txt"
        content_lines = [f"{k}: {str(v) if v is not None else ''}" for k, v in meta.items()]
    else:
        filename = f"{image_id}_no_meta.txt"
        content_lines = [
            "No metadata available.",
            f"URL: https://civitai.com/images/{image_id}?username={username}"
        ]

    try:
        meta_path = safe_path_join(item_dir, filename)
    except ValueError as e:
        logger_md.error(f"Path traversal blocked for meta file {filename}: {e}")
        return

    try:
        with open(meta_path, "w", encoding='utf-8') as f:
            f.write("\n".join(content_lines))
    except OSError as e:
        logger_md.error(f"Error writing metadata file {meta_path}: {e}")


# ============================================================
# Core download function
# ============================================================

def download_file_or_image(url, output_path, username, retry_count=0, max_retries=max_tries,
                           expected_size=0, expected_hash=None):
    """Download a file or image with temp directory safety and HTTP Range resume support.
    
    Returns:
        (bool, str|None): Tuple of (success, error_reason).
        On success: (True, None). On failure: (False, 'error description').
    """
    # Normalize the output filename
    output_dir = os.path.dirname(output_path)
    output_filename = normalize_filename(os.path.basename(output_path))
    output_path = os.path.join(output_dir, output_filename)
    output_path = ensure_path_length(output_path)
    
    is_image = 'image.civitai.com' in url or url.endswith(('.jpg', '.jpeg', '.png', '.webp'))
    short_name = os.path.basename(output_path)[:50]
    
    # Check if file already exists and is valid
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if is_image:
            return True, None
        
        # For model files — verify integrity
        # By default only check file size (instant). SHA256 hash is only
        # verified when --deep_check is enabled, to avoid reading entire
        # multi-GB files on every run.
        if expected_size > 0 or expected_hash:
            check_hash = expected_hash if args.deep_check else None
            if verify_file_integrity(output_path, check_hash, expected_size):
                return True, None
            else:
                tqdm_print(f"  {fmt_user(username)} {fmt_warn(f'{short_name} — integrity check failed, re-downloading')}")
                try:
                    os.remove(output_path)
                except (OSError, PermissionError) as e:
                    logger_md.error(f"Error removing invalid file {output_path}: {e}")
                    return False, f"Cannot remove invalid file: {e}"
        else:
            return True, None  # No data to verify — assume file is valid
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Temp file for safe downloading
    temp_path = get_temp_path(url, output_path)
    
    # Retry loop — replaces recursive calls for cleaner stack traces
    attempt = retry_count
    while attempt <= max_retries:
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Resume: if temp file exists from interrupted download, try to continue
            existing_size = 0
            if os.path.exists(temp_path):
                existing_size = os.path.getsize(temp_path)
                if existing_size > 0:
                    headers['Range'] = f'bytes={existing_size}-'
            
            response = session.get(url, stream=True, timeout=(30, 300), headers=headers)
            
            # Check resume support
            is_resumed = (response.status_code == 206)
            if response.status_code == 416:
                # Range not satisfiable — file already fully downloaded
                if os.path.exists(temp_path):
                    shutil.move(temp_path, output_path)
                    return True, None
            
            response.raise_for_status()
            
            # Detect content type and adjust extension
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type:
                file_extension = '.jpg'
            elif 'video' in content_type:
                file_extension = '.mp4'
            else:
                file_extension = os.path.splitext(output_path)[1]
            
            output_path = os.path.splitext(output_path)[0] + file_extension
            output_path = ensure_path_length(output_path)
            
            # Determine total size
            content_length = int(response.headers.get('content-length', 0))
            if is_resumed:
                total_size = existing_size + content_length
            else:
                total_size = content_length
                existing_size = 0
            
            # Register for progress tracking (model files + images + other downloads)
            # This keeps the "closest to completion" bar active even while downloading
            # previews/examples, which otherwise looks like the script is stuck.
            tracking = False
            if total_size > 0:
                _register_download(short_name, total_size)
                tracking = True
            
            try:
                # Download to temp file with speed monitoring
                write_mode = 'ab' if is_resumed else 'wb'
                downloaded_size = existing_size
                slow_since = None  # timestamp when speed dropped below threshold
                speed_window_bytes = 0
                speed_window_start = time.time()
                
                if tracking:
                    _update_download_progress(existing_size)
                
                with open(temp_path, write_mode) as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            chunk_len = len(chunk)
                            downloaded_size += chunk_len
                            speed_window_bytes += chunk_len
                            
                            if tracking:
                                _update_download_progress(downloaded_size)
                            
                            # Speed monitoring (check every 5 seconds)
                            now = time.time()
                            window_elapsed = now - speed_window_start
                            if window_elapsed >= 5.0:
                                current_speed = speed_window_bytes / window_elapsed
                                speed_window_bytes = 0
                                speed_window_start = now
                                
                                if current_speed < SLOW_SPEED_THRESHOLD:
                                    if slow_since is None:
                                        slow_since = now
                                    elif now - slow_since >= SLOW_SPEED_TIMEOUT:
                                        # Speed below threshold for too long — restart
                                        speed_kb = current_speed / 1024
                                        tqdm_print(f"  {fmt_warn(f'{short_name} — slow ({speed_kb:.0f} KB/s), restarting...')}")
                                        logger_md.warning(f"[{username}] {short_name} - slow download ({speed_kb:.0f} KB/s), restarting")
                                        response.close()
                                        raise requests.exceptions.ConnectionError(f"Slow download restart ({speed_kb:.0f} KB/s)")
                                else:
                                    slow_since = None  # speed recovered
                
                # Verify integrity (for model files) — size-only check after download
                # SHA256 is redundant here: the file was just received over HTTP with
                # its own integrity guarantees. Size check catches truncated downloads.
                if not is_image and expected_size > 0:
                    if not verify_file_integrity(temp_path, None, expected_size):
                        tqdm_print(f"  {fmt_user(username)} {fmt_error(f'{short_name} — integrity check failed')}")
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                        
                        if attempt < max_retries:
                            attempt += 1
                            tqdm_print(f"  {fmt_user(username)} {fmt_warn(f'Retry {attempt + 1}/{max_retries + 1}: {short_name}')}")
                            time.sleep(retry_delay * attempt)
                            continue  # retry via loop
                        else:
                            tqdm_print(f"  {fmt_user(username)} {fmt_error(f'{short_name} — failed after {max_retries + 1} attempts')}")
                            return False, f"Integrity check failed after {max_retries + 1} attempts"
                
                # Success — move from temp to final path
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.move(temp_path, output_path)
                return True, None
            finally:
                if tracking:
                    _unregister_download()
        
        except (requests.exceptions.ConnectionError, ConnectionResetError, requests.exceptions.Timeout) as e:
            # Keep temp file for resume on next attempt
            
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** attempt)
                error_type = "timeout" if isinstance(e, requests.exceptions.Timeout) else "connection"
                tqdm_print(f"  {fmt_user(username)} {fmt_warn(f'{error_type} error, retrying in {wait_time}s: {short_name}')}")
                logger_md.warning(f"[{username}] {short_name} - {error_type} error: {e}")
                time.sleep(wait_time)
                attempt += 1
                continue  # retry via loop
            else:
                tqdm_print(f"  {fmt_user(username)} {fmt_error(f'{short_name} — failed after {max_retries + 1} attempts')}")
                logger_md.error(f"[{username}] {short_name} - final failure: {e}")
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass
                return False, f"Connection failed after {max_retries + 1} attempts: {e}"
        
        except requests.exceptions.HTTPError as e:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            
            status = e.response.status_code if e.response else 'unknown'
            logger_md.error(f"[{username}] {short_name} - HTTP {status}: {e}")
            tqdm_print(f"  {fmt_user(username)} {fmt_error(f'HTTP {status}: {short_name}')}")
            return False, f"HTTP {status}"
        
        except Exception as e:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            logger_md.error(f"[{username}] {short_name} - error: {e}", exc_info=True)
            tqdm_print(f"  {fmt_user(username)} {fmt_error(f'{short_name}: {e}')}")
            return False, f"Error: {e}"
    
    return False, "All retries exhausted"  # All retries exhausted


# ============================================================
# Model file download
# ============================================================

def _is_skipped(path):
    """Check if a folder (or any parent up to OUTPUT_DIR) contains a .skip marker."""
    path = os.path.normpath(path)
    output_norm = os.path.normpath(OUTPUT_DIR)
    while path and path != output_norm and len(path) >= len(output_norm):
        if os.path.isfile(os.path.join(path, SKIP_MARKER)):
            return True
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return False


def download_model_files(username, item_name, model_version, item, download_type, exclude_type, failed_downloads_file):
    """Download all files for one model version."""
    model_id = item['id']
    model_name_with_id = f"{model_id:07d} - {item_name}"
    item_name_sanitized = sanitize_name(model_name_with_id, max_length=MAX_COMPONENT_LENGTH)
    
    primary_category = categorize_item(item)
    base_model = item.get('baseModel')
    
    # Brute-force strip ALL Windows-forbidden chars from every path component.
    # This catches unicode lookalikes (fullwidth pipe ｜, box-drawing │, etc.)
    # that regex character classes may miss.
    _win_forbidden = str.maketrans({
        '<': '_', '>': '_', ':': '_', '"': '_', '/': '_',
        '\\': '_', '|': '_', '?': '_', '*': '_',
        '\uFF5C': '_',  # fullwidth vertical line ｜
        '\u2502': '_',  # box drawings light vertical │
        '\u2503': '_',  # box drawings heavy vertical ┃
        '\u01C0': '_',  # latin letter dental click ǀ
        '\u2223': '_',  # divides ∣
        '\u2016': '_',  # double vertical line ‖
    })
    item_name_sanitized = item_name_sanitized.translate(_win_forbidden)
    primary_category = primary_category.translate(_win_forbidden)

    try:
        username_safe = sanitize_username_for_path(username)
    except ValueError as e:
        logger_md.error(f"Rejected unsafe username for path: {e}")
        return item_name, 0, 0

    version_folder_raw = model_version.get('name', 'Version Unknown')
    version_folder = sanitize_name(version_folder_raw, max_length=MAX_COMPONENT_LENGTH)
    version_folder = version_folder.translate(_win_forbidden)

    try:
        if base_model:
            base_model_safe = base_model.translate(_win_forbidden)
            model_folder = safe_path_join(OUTPUT_DIR, username_safe, primary_category, base_model_safe, item_name_sanitized)
        else:
            model_folder = safe_path_join(OUTPUT_DIR, username_safe, primary_category, item_name_sanitized)
        final_dir = safe_path_join(model_folder, version_folder)
    except ValueError as e:
        logger_md.error(f"Path traversal blocked for {item_name}: {e}")
        return item_name, 0, 0

    # ── Skip marker check ─────────────────────────────────────────
    # If model_folder or final_dir contains .skip — skip entirely.
    if _is_skipped(model_folder):
        tqdm_print(f"  {fmt_dim(f'[SKIP]  {item_name}')}")
        return item_name, 0, 0
    if os.path.exists(final_dir) and _is_skipped(final_dir):
        tqdm_print(f"  {fmt_dim(f'[SKIP]  {item_name} / {version_folder_raw}')}")
        return item_name, 0, 0

    os.makedirs(final_dir, exist_ok=True)
    
    # Determine base_file_name from first file
    base_file_name = None
    files = model_version.get('files', [])
    for file in files:
        file_name = file.get('name', '')
        if file_name:
            base_file_name = os.path.splitext(file_name)[0]
            break
    
    downloaded = False
    file_ok = 0
    file_fail = 0
    
    # Download model files — use sizeKB and hashes from already-fetched API data
    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')
        if not file_name or not file_url:
            continue

        try:
            file_name = sanitize_filename_strict(file_name)
        except ValueError as e:
            logger_md.error(f"Rejected unsafe filename from API: {e}")
            continue

        # Skip training data unless explicitly requested.
        # Some models (e.g. LoRA) may include "Training Data" files in the same version.
        file_type = (file.get('type') or '').strip().lower()
        if file_type == 'training data':
            if exclude_type == 'Training_Data':
                continue
            if download_type not in ('Training_Data', 'All', None):
                continue
        
        # Size and hash from API (no extra request needed!)
        size_kb = file.get('sizeKB', 0)
        expected_size = int(size_kb * 1024) if size_kb else 0
        expected_hash = file.get('hashes', {}).get('SHA256')
        
        file_name_sanitized = sanitize_name(file_name, item_name, max_length=MAX_COMPONENT_LENGTH)
        try:
            file_path = safe_path_join(final_dir, file_name_sanitized)
        except ValueError as e:
            logger_md.error(f"Path traversal blocked for file {file_name}: {e}")
            continue
        file_path = ensure_path_length(file_path)
        
        success, fail_reason = download_file_or_image(
            file_url, file_path, username,
            expected_size=expected_size,
            expected_hash=expected_hash
        )
        if success:
            downloaded = True
            file_ok += 1
        else:
            file_fail += 1
            with _failed_file_lock:
                with open(failed_downloads_file, "a", encoding='utf-8') as f:
                    f.write(f"File: {file_name}\nItem: {item_name}\nURL:  {file_url}\nError: {fail_reason}\n{'─' * 40}\n")
    
    # Download preview image
    preview_filename = ""
    if base_file_name:
        preview_filename = normalize_filename(f"{base_file_name}.preview.jpg")
    else:
        preview_filename = normalize_filename(f"{item_name_sanitized}.preview.jpg")
    preview_path = os.path.join(final_dir, preview_filename)
    preview_url_used = None
    
    if not check_preview_exists(final_dir, base_file_name if base_file_name else item_name_sanitized):
        images = model_version.get('images', [])
        for image in images:
            if image.get("type", "image").lower() == "image":
                preview_url = image.get("url", "")
                if preview_url:
                    p_success, _ = download_file_or_image(preview_url, preview_path, username)
                    if p_success:
                        preview_url_used = preview_url
                        break
    
    # Fetch generation metadata (prompt, sampler, seed, ...) for this version's images
    image_meta_headers = {"Authorization": f"Bearer {token}"}
    image_meta_by_id = fetch_image_metadata(model_version.get('id'), image_meta_headers)

    # Download example images
    images = model_version.get('images', [])
    examples_to_download = []
    for image in images:
        image_url = image.get("url", "")
        if not image_url:
            continue
        if preview_url_used and image_url == preview_url_used:
            continue
        
        image_id = image.get('id', '')
        if not image_id:
            continue
        
        image_id_str = str(image_id)
        
        existing_image = check_existing_image(image_id_str, os.path.join(OUTPUT_DIR, username_safe))
        if existing_image:
            continue
        
        image_filename_raw = f"{item_name}_{image_id}.jpeg"
        image_filename_sanitized = sanitize_name(image_filename_raw, item_name, max_length=MAX_COMPONENT_LENGTH)
        examples_dir = os.path.join(final_dir, "examples")
        image_path = os.path.join(examples_dir, image_filename_sanitized)
        
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            continue
        
        examples_to_download.append((image_url, image_path, image_id_str))
    
    if examples_to_download:
        examples_dir = os.path.join(final_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        for image_url, image_path, image_id_str in examples_to_download:
            success, fail_reason = download_file_or_image(image_url, image_path, username)
            if success:
                downloaded = True
                file_ok += 1
                add_to_image_cache(image_id_str, image_path)
                meta_key = int(image_id_str) if image_id_str.isdigit() else None
                meta = image_meta_by_id.get(meta_key) if meta_key else None
                write_image_meta_file(meta, image_id_str, examples_dir, username)
            else:
                file_fail += 1
                with _failed_file_lock:
                    with open(failed_downloads_file, "a", encoding='utf-8') as f:
                        f.write(f"Image: {os.path.basename(image_path)}\nItem: {item_name}\nURL:  {image_url}\nError: {fail_reason}\n{'─' * 40}\n")
    
    # Save info files — store only the current version to avoid
    # cross-version size/hash mismatches during integrity checks
    if base_file_name:
        info_filename = normalize_filename(f"{base_file_name}.civitai.info")
    else:
        info_filename = normalize_filename(f"{item_name_sanitized}.civitai.info")
    info_path = os.path.join(final_dir, info_filename)
    item_single_version = item.copy()
    item_single_version['modelVersions'] = [model_version]
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(item_single_version, f, indent=4)
    
    if base_file_name:
        basejson_filename = normalize_filename(f"{base_file_name}.json")
    else:
        basejson_filename = normalize_filename(f"{item_name_sanitized}.json")
    basejson_path = os.path.join(final_dir, basejson_filename)
    clean_description = re.sub(r'<[^>]*>', '', item.get('description', '') or '')
    trigger_words = model_version.get('trainedWords', [])
    if isinstance(trigger_words, list):
        notes = ", ".join(trigger_words)
    else:
        notes = str(trigger_words)
    basejson_data = {
        "description": clean_description,
        "notes": notes
    }
    with open(basejson_path, "w", encoding="utf-8") as f:
        json.dump(basejson_data, f, indent=4)
    
    # triggerWords.txt — one word per line
    trigger_words = model_version.get('trainedWords', [])
    if trigger_words:
        trigger_file_path = os.path.join(final_dir, "triggerWords.txt")
        with open(trigger_file_path, "w", encoding="utf-8") as f:
            if isinstance(trigger_words, list):
                for word in trigger_words:
                    f.write(f"{word}\n")
            else:
                f.write(str(trigger_words) + "\n")
    
    # Final duplicate check
    find_and_remove_duplicates_in_directory(final_dir)
    
    return item_name, file_ok, file_fail


# ============================================================
# User processing
# ============================================================

def process_username(username, download_type, exclude_type=None, base_models=None):
    """Process a username and download the specified type of content."""

    try:
        username_safe = sanitize_username_for_path(username)
    except ValueError as e:
        print(fmt_error(f"Invalid username: {e}"))
        return

    user_folder = safe_path_join(OUTPUT_DIR, username_safe)
    if _is_skipped(user_folder):
        print(fmt_info(f"[{username}] User completely skipped due to .skip marker!"))
        return

    failed_downloads_file = os.path.join(LOGS_DIR, f"failed_downloads_{username_safe}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")
    
    params = {
        "username": username,
        "limit": 100,
        "nsfw": "true"
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    next_page = url
    downloaded_items = set()
    total_items = 0
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    items_processed = 0
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        first_page = True
        
        while next_page:
            try:
                response = session.get(next_page, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Get totalItems from first page (single request, no double-fetch)
                if first_page:
                    total_items = data.get('metadata', {}).get('totalItems', 0)
                    tqdm_print(f"  {Style.DIM}{username}: {total_items} items found{Style.RESET}")
                    first_page = False
                
                items = data.get('items', [])
                if not items:
                    break
                
                for item in items:
                    item_category = categorize_item(item)
                    
                    # Check download_type filter
                    if download_type is not None and download_type != 'All':
                        if item_category != download_type:
                            skipped_count += 1
                            items_processed += 1
                            continue
                    
                    # Check exclude_type filter
                    if exclude_type is not None:
                        if item_category == exclude_type:
                            skipped_count += 1
                            items_processed += 1
                            continue
                    
                    # Check base model filter
                    if base_models is not None:
                        model_versions = item.get('modelVersions', [])
                        has_matching = any(
                            v.get('baseModel') in base_models for v in model_versions
                        )
                        if not has_matching:
                            skipped_count += 1
                            items_processed += 1
                            continue
                    
                    item_name = item['name']
                    if item_name not in downloaded_items:
                        downloaded_items.add(item_name)
                        for version in item.get('modelVersions', []):
                            version_base_model = version.get('baseModel')
                            if base_models is not None and version_base_model not in base_models:
                                continue
                            
                            item_with_base_model = item.copy()
                            item_with_base_model['baseModel'] = version_base_model
                            future = executor.submit(
                                download_model_files,
                                username,
                                item_name,
                                version,
                                item_with_base_model,
                                download_type,
                                exclude_type,
                                failed_downloads_file
                            )
                            futures.append(future)
                    
                    items_processed += 1
                
                metadata = data.get('metadata', {})
                next_page = validate_next_page_url(metadata.get('nextPage'))

            except Exception as e:
                tqdm_print(fmt_error(f"Error processing page: {e}"))
                logger_md.error(f"Error processing page for {username}: {e}", exc_info=True)
                break
        
        # Wait for all downloads — two progress bars:
        # 1) Overall: completed models / total models
        # 2) File: the download closest to completion (auto-switches)
        if futures:
            total_tasks = len(futures)
            overall_bar = tqdm(
                total=total_tasks, position=1, leave=False,
                desc=f"  {Style.BOLD}{username}{Style.RESET}",
                bar_format='{desc}: {percentage:3.0f}%|{bar:25}| {n}/{total} models [{elapsed}<{remaining}, {rate_fmt}]',
                mininterval=0.3
            )
            file_bar = tqdm(
                total=100, position=0, leave=False,
                desc=f"  {Style.DIM}waiting...{Style.RESET}",
                bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{rate_fmt} ETA {remaining}]',
                unit='B', unit_scale=True, mininterval=0.3
            )
            
            remaining_futures = set(futures)
            while remaining_futures:
                done, remaining_futures = wait(remaining_futures, timeout=0.3, return_when=FIRST_COMPLETED)
                
                for future in done:
                    try:
                        item_name, ok_count, fail_count_result = future.result()
                        downloaded_count += ok_count
                        failed_count += fail_count_result
                    except Exception as e:
                        logger_md.error(f"Download task error: {e}")
                        failed_count += 1
                        # Log unhandled task exceptions to file
                        with _failed_file_lock:
                            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                                f.write(f"Task Error: {e}\n{'─' * 40}\n")
                    overall_bar.update(1)
                
                # Update file bar — show the download closest to completion
                with _active_downloads_lock:
                    if _active_downloads:
                        best = max(
                            _active_downloads.values(),
                            key=lambda d: (d['downloaded'] / d['total']) if d['total'] > 0 else 0
                        )
                        file_bar.total = best['total']
                        file_bar.n = best['downloaded']
                        file_bar.set_description(f"  {best['name'][:45]}")
                        file_bar.refresh()
                    else:
                        file_bar.total = 100
                        file_bar.n = 0
                        file_bar.set_description(f"  {Style.DIM}waiting...{Style.RESET}")
                        file_bar.refresh()
            
            overall_bar.close()
            file_bar.close()
            # Clear the two bar lines
            print("\033[K", end="")
            print("\033[K", end="")
    
    # Remove empty failed_downloads file if no failures
    if failed_count == 0:
        try:
            os.remove(failed_downloads_file)
        except OSError:
            pass
    
    # Compact user result box
    lines = []
    if downloaded_count > 0:
        lines.append(f"{Style.GREEN}✓{Style.RESET} {downloaded_count} files downloaded")
    elif total_items == 0 and downloaded_count == 0 and failed_count == 0:
        lines.append(f"{Style.GREEN}✓{Style.RESET} up to date")
    if skipped_count > 0:
        lines.append(f"{Style.DIM}↷ {skipped_count} skipped{Style.RESET}")
    if failed_count > 0:
        lines.append(f"{Style.RED}✗ {failed_count} failed{Style.RESET}  {Style.DIM}(see logs/failed_downloads_{username}.txt){Style.RESET}")
    
    if not lines:
        lines.append(f"{Style.GREEN}✓{Style.RESET} nothing to do")
    
    ui_user_box(username, lines)

    return total_items, downloaded_count, skipped_count, failed_count


def fetch_model_by_id(model_id, headers):
    """Fetch a single model by ID from the CivitAI API.

    Returns:
        tuple: (model data dict, error message or None)
    """
    url = f"{BASE_URL}/{model_id}"
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 'unknown'
        if e.response is not None and e.response.status_code == 404:
            return None, f"Model {model_id} not found."
        return None, f"HTTP error {status} fetching model {model_id}."
    except requests.exceptions.RequestException as e:
        logger_md.error(f"Network error fetching model {model_id}: {type(e).__name__}")
        return None, f"Network error fetching model {model_id}."
    except ValueError:
        return None, f"Invalid JSON response for model {model_id}."


def process_model_ids(model_ids, download_type, exclude_type=None, base_models=None):
    """Fetch and download specific models by their numeric IDs, bypassing username search."""
    headers = {"Authorization": f"Bearer {token}"}

    failed_downloads_file = os.path.join(LOGS_DIR, "failed_downloads_by_id.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write("Failed Downloads for Model IDs\n\n")

    grand_downloaded = 0
    grand_failed = 0

    for model_id in model_ids:
        tqdm_print(f"\nFetching model {model_id}...")
        item, error = fetch_model_by_id(model_id, headers)
        if error:
            tqdm_print(fmt_error(f"  {error}"))
            continue

        item_name = item.get('name')
        if not item_name or not isinstance(item_name, str):
            tqdm_print(fmt_error(f"  Skipping model {model_id}: invalid name"))
            continue

        creator = item.get('creator') or {}
        username = creator.get('username', 'unknown_user')
        tqdm_print(f"  Model: {item_name} (by {username})")

        model_versions = item.get('modelVersions', [])
        if not model_versions:
            tqdm_print(fmt_warn(f"  No versions found for model {model_id}"))
            continue

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for version in model_versions:
                version_base_model = version.get('baseModel')
                if base_models is not None and not base_model_matches(version_base_model, base_models):
                    continue

                item_with_base_model = item.copy()
                item_with_base_model['baseModel'] = version_base_model
                future = executor.submit(
                    download_model_files,
                    username,
                    item_name,
                    version,
                    item_with_base_model,
                    download_type,
                    exclude_type,
                    failed_downloads_file
                )
                futures.append(future)

            for future in tqdm(futures, desc=f"  Downloading {item_name}", unit="version", leave=False):
                try:
                    _, ok_count, fail_count_result = future.result()
                    grand_downloaded += ok_count
                    grand_failed += fail_count_result
                except Exception as e:
                    logger_md.error(f"Download task error for model {model_id}: {e}")
                    grand_failed += 1

    if grand_failed == 0:
        try:
            os.remove(failed_downloads_file)
        except OSError:
            pass

    tqdm_print(f"\n{fmt_ok(f'Downloaded: {grand_downloaded}')}" + (f"  {fmt_error(f'Failed: {grand_failed}')}" if grand_failed else ""))
    return grand_downloaded, grand_failed


def search_for_training_data_files(item):
    """Search for files with type 'Training Data' in the model versions."""
    training_data_files = []
    model_versions = item.get("modelVersions", [])
    for version in model_versions:
        for file in version.get("files", []):
            if file.get("type") == "Training Data":
                training_data_files.append(file.get("name", ""))
    return training_data_files


def fetch_all_models(token, username):
    """Fetch and categorize all models for a username."""
    categorized_items = {
        'Checkpoints': [],
        'Embeddings': [],
        'Lora': [],
        'Training_Data': [],
        'Other': []
    }
    other_item_types = []
    
    # Token via header, not in URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    next_page = f"{BASE_URL}?username={username}&nsfw=true"
    first_next_page = None
    iteration_count = 0
    
    while next_page:
        response = session.get(next_page, headers=headers)
        data = response.json()
        for item in data.get("items", []):
            try:
                category = categorize_item(item)
                categorized_items[category].append(item.get("name", ""))
                training_data_files = search_for_training_data_files(item)
                if training_data_files:
                    categorized_items['Training_Data'].extend(training_data_files)
                if category == 'Other':
                    other_item_types.append((item.get("name", ""), item.get("type", None)))
            except Exception as e:
                logger_md.error(f"Error categorizing item: {item} - {e}")
        metadata = data.get('metadata', {})
        next_page = validate_next_page_url(metadata.get('nextPage'))
        if first_next_page is None:
            first_next_page = next_page
        
        iteration_count += 1
        
        if next_page and next_page == first_next_page and iteration_count > 1:
            logger_md.error("Termination condition met: first nextPage URL repeated.")
            break
        elif not next_page or not metadata:
            break
    
    total_count = sum(len(items) for items in categorized_items.values())
    summary_file_path = os.path.join(LOGS_DIR, f"{username}.txt")
    with open(summary_file_path, "w", encoding='utf-8') as file:
        file.write("Summary:\n")
        file.write(f"Total - Count: {total_count}\n")
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
        file.write("\nDetailed Listing:\n")
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
            if category == 'Other':
                for item_name, item_type in other_item_types:
                    file.write(f"{category} - Item: {item_name} - Type: {item_type}\n")
            else:
                for item_name in items:
                    file.write(f"{category} - Item: {item_name}\n")
            file.write("\n")
    return categorized_items


# ============================================================
# Broken file scanner and fixer (parallelized)
# ============================================================

def _check_single_file(entry, deep_check=False):
    """Check integrity of a single file. Returns (entry, is_broken, reason) tuple.
    
    By default only checks file size (instant). With deep_check=True also
    verifies SHA256 hash (slow — reads entire file)."""
    local_path = entry['local_path']
    expected_size = entry['expected_size']
    expected_hash = entry['expected_hash']
    ext = entry['ext']
    
    if not os.path.exists(local_path):
        return entry, False, None  # File gone — skip
    
    file_size = os.path.getsize(local_path)
    
    # Check 1: Empty file
    if file_size == 0:
        return entry, True, "empty"
    
    # Check 2: Too small for model files (< 1MB suspicious)
    if ext.lower() in ('.safetensors', '.ckpt') and file_size < 1024 * 1024:
        return entry, True, f"too small ({file_size:,} bytes)"
    
    # Check 3: Size mismatch (> 2% difference)
    if expected_size > 0:
        size_diff = abs(file_size - expected_size)
        if size_diff > expected_size * 0.02:
            return entry, True, f"size mismatch ({file_size:,} vs {expected_size:,})"
    
    # Check 4: SHA256 mismatch — ONLY with --deep_check flag (very slow for large files)
    if deep_check and expected_hash and expected_size > 0:
        actual_hash = get_file_sha256(local_path)
        if actual_hash and actual_hash != expected_hash.upper():
            return entry, True, "hash mismatch"
    
    return entry, False, None


def _fix_single_file(entry):
    """Re-download a single broken file. Runs in a thread pool."""
    local_path = entry['local_path']
    normalized_name = entry['normalized_name']
    download_url = entry['download_url']
    expected_size = entry['expected_size']
    expected_hash = entry['expected_hash']
    dirpath = entry['dirpath']
    
    # Remove broken file
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except OSError as e:
        logger_md.error(f"Cannot remove broken file {local_path}: {e}")
        return entry, False
    
    # Determine username from path
    username = "repair"
    rel_path = os.path.relpath(dirpath, OUTPUT_DIR)
    parts = rel_path.split(os.sep)
    if parts:
        username = parts[0]
    
    target_path = os.path.join(dirpath, normalized_name)
    success, _ = download_file_or_image(
        download_url, target_path, username,
        expected_size=expected_size,
        expected_hash=expected_hash
    )
    return entry, success


def scan_and_fix_broken_files(root_dir, deep_check=False):
    """Scan existing directories, check model files against .civitai.info data,
    and re-download any broken or corrupted files.
    
    Args:
        root_dir: Root directory to scan.
        deep_check: If True, verify SHA256 hashes (slow). Default: size-only (instant).
    """
    if not os.path.exists(root_dir):
        return
    
    model_extensions = {'.safetensors', '.ckpt', '.pt', '.bin', '.pth', '.zip', '.onnx'}
    
    # ── Phase 1: Collect all files to check ──
    candidates = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        info_files = [f for f in filenames if f.endswith('.civitai.info')]
        if not info_files:
            continue
        
        info_path = os.path.join(dirpath, info_files[0])
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                item_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger_md.warning(f"Cannot read {info_path}: {e}")
            continue
        
        # Determine which version this directory belongs to by matching
        # the directory name against version names from the API data.
        # This prevents cross-version mismatches when multiple versions
        # share the same filename but have different sizes/hashes.
        dir_basename = os.path.basename(dirpath)
        
        matched_version = None
        for version in item_data.get('modelVersions', []):
            version_name = version.get('name', '')
            sanitized_version_name = sanitize_name(version_name, max_length=MAX_COMPONENT_LENGTH)
            if dir_basename == sanitized_version_name or dir_basename == version_name:
                matched_version = version
                break
        
        if matched_version:
            # Use only the files from the matched version
            api_files = matched_version.get('files', [])
        else:
            # Fallback: could not match directory to a version — use all files
            # (this handles legacy directories or renamed folders)
            api_files = []
            for version in item_data.get('modelVersions', []):
                for file_entry in version.get('files', []):
                    api_files.append(file_entry)
        
        if not api_files:
            continue
        
        checked_paths = set()
        
        for api_file in api_files:
            api_name = api_file.get('name', '')
            size_kb = api_file.get('sizeKB', 0)
            expected_size = int(size_kb * 1024) if size_kb else 0
            expected_hash = api_file.get('hashes', {}).get('SHA256')
            download_url = api_file.get('downloadUrl', '')
            
            if not api_name or not download_url:
                continue
            
            _, ext = os.path.splitext(api_name)
            if ext.lower() not in model_extensions:
                continue
            
            sanitized_name = sanitize_name(api_name, max_length=MAX_COMPONENT_LENGTH)
            normalized_name = normalize_filename(sanitized_name)
            
            local_path = None
            for candidate in [normalized_name, sanitized_name, api_name]:
                candidate_path = os.path.join(dirpath, candidate)
                if os.path.exists(candidate_path) and candidate_path not in checked_paths:
                    local_path = candidate_path
                    break
            
            if local_path is None:
                continue
            
            checked_paths.add(local_path)
            
            candidates.append({
                'local_path': local_path,
                'normalized_name': normalized_name,
                'download_url': download_url,
                'expected_size': expected_size,
                'expected_hash': expected_hash,
                'ext': ext,
                'dirpath': dirpath,
            })
    
    if not candidates:
        print(fmt_ok("No model files to check"))
        return
    
    check_mode = "deep (SHA256)" if deep_check else "quick (size)"
    print(fmt_info(f"Checking {len(candidates)} model file(s) — {check_mode} mode"))
    
    # ── Phase 2: Parallel integrity checks ──
    broken_files = []
    check_threads = min(max_threads, len(candidates), 8)  # Cap at 8 for I/O
    
    with ThreadPoolExecutor(max_workers=check_threads) as executor:
        progress = tqdm(
            total=len(candidates), desc="Integrity check",
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n}/{total} [{elapsed}<{remaining}]',
            leave=False
        )
        
        futures = {executor.submit(_check_single_file, entry, deep_check): entry for entry in candidates}
        
        for future in futures:
            try:
                entry, is_broken, reason = future.result()
                if is_broken:
                    fname = os.path.basename(entry['local_path'])
                    broken_files.append(entry)
                    tqdm.write(f"  {fmt_error(f'BROKEN ({reason}): {fname}')}")
            except Exception as e:
                logger_md.error(f"Error checking file: {e}")
            progress.update(1)
        
        progress.close()
    
    if not broken_files:
        print(fmt_ok(f"All {len(candidates)} file(s) passed integrity check"))
        return
    
    # ── Phase 3: Parallel re-downloads ──
    print(fmt_warn(f"Found {len(broken_files)} broken file(s), re-downloading in parallel..."))
    
    fixed_count = 0
    failed_count = 0
    dl_threads = min(max_threads, len(broken_files))
    
    with ThreadPoolExecutor(max_workers=dl_threads) as executor:
        progress = tqdm(
            total=len(broken_files), desc="Repairing",
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n}/{total} [{elapsed}<{remaining}]',
            leave=False
        )
        
        futures = {executor.submit(_fix_single_file, entry): entry for entry in broken_files}
        
        for future in futures:
            try:
                entry, success = future.result()
                fname = os.path.basename(entry['normalized_name'])
                if success:
                    fixed_count += 1
                    tqdm.write(f"  {fmt_ok(f'FIXED: {fname}')}")
                else:
                    failed_count += 1
                    tqdm.write(f"  {fmt_error(f'FAILED: {fname}')}")
            except Exception as e:
                failed_count += 1
                logger_md.error(f"Error fixing file: {e}")
            progress.update(1)
        
        progress.close()
    
    print(f"  {fmt_warn(f'Broken: {len(broken_files)} | Fixed: {fixed_count} | Failed: {failed_count}')}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    _main_start = time.time()

    # ── Banner ──
    ui_banner()

    if model_ids:
        # Model-ID mode bypasses username search and the per-user maintenance
        # phases (migration/dedup/integrity scan are keyed by username
        # directories we don't know ahead of fetching each model).
        ui_config([f"model:{m}" for m in model_ids], download_type, exclude_type, base_models)
        cleanup_temp_dir()
        process_model_ids(model_ids, download_type, exclude_type, base_models)
        elapsed = time.time() - _main_start
        print(f"\n  {Style.DIM}Time: {int(elapsed // 60)}m {int(elapsed % 60)}s{Style.RESET}")
        print(f"\n  {Style.GREEN}{Style.BOLD}✓ All done!{Style.RESET}\n")
        sys.exit(0)

    ui_config(usernames, download_type, exclude_type, base_models)

    TOTAL_PHASES = 5

    # Clean up temp directory from previous interrupted downloads
    cleanup_temp_dir()
    
    # ── Phase 1: Migration ──
    ui_phase(1, TOTAL_PHASES, "Directory Migration", status='active')
    for username in usernames:
        user_dir = os.path.join(OUTPUT_DIR, username)
        if os.path.exists(user_dir):
            migrate_long_directories(user_dir)
    ui_phase(1, TOTAL_PHASES, "Directory Migration", status='done', detail="complete")
    
    # ── Phase 2: Duplicate Cleanup ──
    ui_phase(2, TOTAL_PHASES, "Duplicate Cleanup", status='active')
    total_dup_removed = 0
    for username in usernames:
        user_dir = os.path.join(OUTPUT_DIR, username)
        if os.path.exists(user_dir):
            removed_count = find_and_remove_duplicates_in_directory(user_dir)
            total_dup_removed += removed_count
            
            duplicates = find_image_duplicates(user_dir)
            if duplicates:
                img_removed = 0
                for image_id, paths in duplicates.items():
                    files_with_model_name = []
                    files_without_model_name = []
                    
                    for path in paths:
                        filename = os.path.basename(path)
                        if has_model_name(filename):
                            files_with_model_name.append(path)
                        else:
                            files_without_model_name.append(path)
                    
                    if files_with_model_name:
                        for path in files_without_model_name:
                            try:
                                os.remove(path)
                                img_removed += 1
                            except Exception as e:
                                logger_md.debug(f"Error removing {path}: {e}")
                        
                        for path in files_with_model_name[1:]:
                            try:
                                os.remove(path)
                                img_removed += 1
                            except Exception as e:
                                logger_md.debug(f"Error removing {path}: {e}")
                    else:
                        for path in paths[1:]:
                            try:
                                os.remove(path)
                                img_removed += 1
                            except Exception as e:
                                logger_md.debug(f"Error removing {path}: {e}")
                total_dup_removed += img_removed
    
    dup_detail = f"removed {total_dup_removed}" if total_dup_removed > 0 else "no duplicates"
    ui_phase(2, TOTAL_PHASES, "Duplicate Cleanup", status='done', detail=dup_detail)
    
    # ── Phase 3: Integrity Check (background thread) ──
    integrity_error = [None]
    integrity_result = [0, 0]  # [checked, broken]
    
    def _integrity_check_worker():
        try:
            for uname in usernames:
                udir = os.path.join(OUTPUT_DIR, uname)
                scan_and_fix_broken_files(udir, deep_check=args.deep_check)
        except Exception as e:
            integrity_error[0] = e
            logger_md.error(f"Integrity check error: {e}", exc_info=True)
    
    check_mode = "SHA256" if args.deep_check else "size"
    ui_phase(3, TOTAL_PHASES, "Integrity Check", status='active', detail=f"mode: {check_mode}")
    integrity_thread = threading.Thread(target=_integrity_check_worker, daemon=True)
    integrity_thread.start()
    
    # ── Phase 4: Downloading ──
    ui_phase(4, TOTAL_PHASES, "Downloading", status='active', detail=f"{len(usernames)} users")
    print()
    
    grand_total = 0
    grand_downloaded = 0
    grand_skipped = 0
    grand_failed = 0
    
    for username in usernames:
        t, d, s, f = process_username(username, download_type, exclude_type, base_models)
        grand_total += t
        grand_downloaded += d
        grand_skipped += s
        grand_failed += f
    
    print()
    dl_detail = f"{grand_downloaded} downloaded"
    if grand_failed > 0:
        dl_detail += f", {grand_failed} failed"
    ui_phase(4, TOTAL_PHASES, "Downloading", status='done' if grand_failed == 0 else 'warn', detail=dl_detail)
    
    # Wait for integrity check to finish
    integrity_thread.join()
    if integrity_error[0]:
        ui_phase(3, TOTAL_PHASES, "Integrity Check", status='fail', detail=str(integrity_error[0]))
    else:
        ui_phase(3, TOTAL_PHASES, "Integrity Check", status='done')
    
    # ── Phase 5: Final Cleanup ──
    ui_phase(5, TOTAL_PHASES, "Final Cleanup", status='active')
    final_removed = 0
    for username in usernames:
        user_dir = os.path.join(OUTPUT_DIR, username)
        if os.path.exists(user_dir):
            removed_count = find_and_remove_duplicates_in_directory(user_dir)
            final_removed += removed_count
    
    cleanup_temp_dir()
    clean_detail = f"removed {final_removed}" if final_removed > 0 else "clean"
    ui_phase(5, TOTAL_PHASES, "Final Cleanup", status='done', detail=clean_detail)
    
    # ── Final Summary ──
    elapsed = time.time() - _main_start
    ui_summary(grand_total, grand_downloaded, grand_skipped, grand_failed, elapsed)
    print(f"\n  {Style.GREEN}{Style.BOLD}✓ All done!{Style.RESET}\n")