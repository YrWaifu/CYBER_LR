import csv
import json
import sys
import time
from pathlib import Path

import requests


# Конфигурация по умолчанию
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
PROMPTS_FILE = "prompts.json"
REPORT_FILE = "inference_report.csv"
REQUEST_TIMEOUT = 180  # секунд на один запрос


def load_prompts(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Файл с запросами не найден: {path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(
            f"Ожидался JSON-массив строк в файле {path}, получено: {type(data).__name__}"
        )

    return data


def check_server(url: str = OLLAMA_URL) -> bool:
    base_url = url.rsplit("/api/", 1)[0]
    try:
        response = requests.get(base_url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def query_ollama(prompt: str, model: str = MODEL_NAME, url: str = OLLAMA_URL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.Timeout:
        return f"[ERROR] Таймаут запроса ({REQUEST_TIMEOUT}s)"
    except requests.RequestException as e:
        return f"[ERROR] Ошибка HTTP-запроса: {e}"
    except ValueError as e:
        return f"[ERROR] Ошибка декодирования JSON-ответа: {e}"


def save_report(rows: list[tuple[str, str]], path: str = REPORT_FILE) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["prompt", "response"])
        writer.writerows(rows)


def run_inference(prompts_path: str = PROMPTS_FILE, report_path: str = REPORT_FILE) -> None:
    print(f"=== Инференс модели {MODEL_NAME} через {OLLAMA_URL} ===\n")

    if not check_server():
        print(
            "[!] Сервер Ollama недоступен. "
            "Убедитесь, что запущена команда `ollama serve`.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    print(f"Загружено {len(prompts)} запросов из {prompts_path}\n")

    rows: list[tuple[str, str]] = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] Запрос: {prompt}")
        t0 = time.time()
        answer = query_ollama(prompt)
        dt = time.time() - t0
        print(f"    -> ответ ({dt:.1f}s): {answer[:120]}{'...' if len(answer) > 120 else ''}\n")
        rows.append((prompt, answer))

    save_report(rows, report_path)
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    run_inference()
