import os
import re
import csv
import json
import requests
from git import Repo

REPOS_DIR = "java_repos"
TXT_OUTPUT = "singleton_dataset_java.txt"
JSON_OUTPUT = "singleton_dataset_java.json"
CSV_OUTPUT = "singleton_dataset_java.csv"

def search_github_repos(query="singleton java", max_repos=5):
    print("🔍 Buscando repositorios en GitHub...")
    url = f"https://api.github.com/search/repositories?q={query}+language:Java&sort=stars&per_page={max_repos}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    repos = response.json().get("items", [])
    return [repo["clone_url"] for repo in repos]

def clone_repo(clone_url, dest_dir):
    if not os.path.exists(dest_dir):
        Repo.clone_from(clone_url, dest_dir)
        print(f"✅ Clonado: {clone_url}")
    else:
        print(f"⚠️ Ya existe: {clone_url}")

def find_singleton_java_code(repo_dir, repo_name):
    singleton_entries = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".java"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        if is_singleton(code):
                            entry = {
                                "repo_name": repo_name,
                                "file_path": path,
                                "class_name": get_class_name(code),
                                "code": code
                            }
                            singleton_entries.append(entry)
                except Exception as e:
                    print(f"Error leyendo archivo: {path} -> {e}")
    return singleton_entries

def is_singleton(code):
    return (
        "private static" in code and
        "getInstance" in code and
        re.search(r"private\s+\w+\s*\(\)", code)
    )

def get_class_name(code):
    match = re.search(r"\bclass\s+(\w+)", code)
    return match.group(1) if match else "Unknown"

def save_txt(singletons):
    with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
        for i, entry in enumerate(singletons):
            f.write(f"// Singleton #{i+1} - {entry['class_name']}\n")
            f.write(f"// Repo: {entry['repo_name']}\n")
            f.write(f"// File: {entry['file_path']}\n")
            f.write(entry["code"] + "\n\n")
    print(f"💾 Dataset TXT guardado en: {TXT_OUTPUT}")

def save_json(singletons):
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(singletons, f, indent=4, ensure_ascii=False)
    print(f"💾 Dataset JSON guardado en: {JSON_OUTPUT}")

def save_csv(singletons):
    with open(CSV_OUTPUT, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["repo_name", "file_path", "class_name", "code"])
        writer.writeheader()
        for entry in singletons:
            writer.writerow(entry)
    print(f"💾 Dataset CSV guardado en: {CSV_OUTPUT}")

def main():
    os.makedirs(REPOS_DIR, exist_ok=True)
    repos = search_github_repos(max_repos=5)
    all_singletons = []

    for repo_url in repos:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        local_path = os.path.join(REPOS_DIR, repo_name)

        try:
            clone_repo(repo_url, local_path)
            singletons = find_singleton_java_code(local_path, repo_name)
            print(f"🔎 {len(singletons)} posibles Singletons en {repo_name}")
            all_singletons.extend(singletons)
        except Exception as e:
            print(f"❌ Error procesando {repo_url}: {e}")

    save_txt(all_singletons)
    save_json(all_singletons)
    save_csv(all_singletons)

if __name__ == "__main__":
    main()
