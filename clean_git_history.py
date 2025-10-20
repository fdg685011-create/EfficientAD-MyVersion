import os
import subprocess

# ==== 基础设置 ====
PROJECT_PATH = r"D:\PythonProject\EfficientAD"
GIT_FILTER_REPO_PATH = r"C:\Program Files\Git\mingw64\libexec\git-core\git-filter-repo"

# 要永久清除历史的文件夹
REMOVE_PATHS = ["datasets", "models", "results"]

print("=====================================================")
print("🧹 一键清理 Git 历史（极简版）")
print("=====================================================")
os.chdir(PROJECT_PATH)

# ==== 自动添加未提交文件 ====
print("🪶 自动提交未保存的更改 ...")
subprocess.run(["git", "add", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["git", "commit", "-m", "auto commit before cleaning"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==== 直接执行 filter-repo（强制模式）====
cmd = ["python", GIT_FILTER_REPO_PATH, "--force"]
for path in REMOVE_PATHS:
    cmd += ["--path", path]
cmd += ["--invert-paths"]

print("🚀 正在执行历史清理（强制模式）...")
subprocess.run(cmd, check=True)

# ==== 清理 Git 缓存 ====
print("🧩 清理 Git 缓存 ...")
subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"])
subprocess.run(["git", "gc", "--prune=now", "--aggressive"])

# ==== 推送远程 ====
print("🌍 强制推送到远程 GitHub 仓库 ...")
subprocess.run(["git", "push", "origin", "--force", "--all"])
subprocess.run(["git", "push", "origin", "--force", "--tags"])

print("=====================================================")
print("🎉 完成！Git 历史已清理并推送。")
print("=====================================================")
