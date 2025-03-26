# **Git 使用规则文档（简单易懂版）**  

本项目使用 **Git** 进行版本控制，以下是基本的 Git 规范，确保团队协作顺畅、代码管理清晰。  

---

## **📌 1. Git 基础概念**
1. **main 分支**（主分支）：始终保持稳定，**不能直接修改**，必须通过 **分支合并** 更新代码。  
2. **开发分支**（feature 分支）：每个功能都应该**单独创建分支**，完成后合并到 `main`。  
3. **临时分支**（hotfix 分支）：紧急修复 bug 时创建，修复后合并回 `main`。  

---

## **📌 2. 克隆项目（新成员加入）**
新成员第一次使用时，需要克隆项目：  
```bash
git clone <项目仓库地址>
cd log_analysis_project  # 进入项目目录
git checkout main        # 切换到主分支
git pull origin main     # 确保是最新版本
```

---

## **📌 3. 创建新分支**
每个开发任务都要**新建一个分支**，命名方式：`feature-模块名` 或 `fix-问题描述`。  
```bash
git checkout main                 # 先切换到主分支
git pull origin main               # 确保 main 是最新的
git checkout -b feature-log-query  # 创建新分支（日志查询功能）
```

---

## **📌 4. 提交代码**
每次修改代码后，都要**提交到自己的分支**：  
```bash
git add .                           # 添加所有修改文件
git commit -m "完成日志查询 API"     # 添加提交说明
git push origin feature-log-query   # 推送到远程仓库
```

---

## **📌 5. 合并代码到 main**
### **✅ 方式 1：本地合并**
1. **切换到 `main` 分支**：
   ```bash
   git checkout main
   git pull origin main  # 获取最新代码，防止冲突
   ```
2. **合并分支**：
   ```bash
   git merge feature-log-query
   ```
3. **解决冲突**（如果有冲突，Git 会提示需要手动修改）：
   - 使用 VSCode 或其他编辑器手动修改冲突文件。  
   - 确保代码正确后，执行：
     ```bash
     git add .
     git commit -m "解决合并冲突"
     ```
4. **推送到远程仓库**：
   ```bash
   git push origin main
   ```

### **✅ 方式 2：使用 Pull Request（推荐）**
**适用于多人协作，避免直接修改 `main`**：
1. **推送自己的分支**：
   ```bash
   git push origin feature-log-query
   ```
2. **在 GitHub / GitLab 上创建 Pull Request**：
   - 选择 `feature-log-query` 分支，合并到 `main`。
   - 让其他人审核后再合并。

---

## **📌 6. 删除分支**
当功能开发完成，分支合并后，删除本地 & 远程分支：
```bash
git branch -d feature-log-query       # 删除本地分支
git push origin --delete feature-log-query  # 删除远程分支
```

---

## **📌 7. 处理多人协作时的冲突**
如果 `git push` 时提示 **"rejected"**，说明 `main` 分支更新了，需要先同步代码：
```bash
git checkout main       # 切换到主分支
git pull origin main    # 拉取最新代码
git checkout feature-log-query  # 回到自己的分支
git merge main         # 把最新代码合并到当前分支
```
如果发生冲突，手动修改冲突文件，再执行：
```bash
git add .
git commit -m "解决冲突"
git push origin feature-log-query
```

---

## **📌 8. Git 工作流程总结**
```bash
# 1. 从主分支创建新分支
git checkout main
git pull origin main
git checkout -b feature-xxx

# 2. 开发 & 提交代码
git add .
git commit -m "完成功能"
git push origin feature-xxx

# 3. 合并到 main
git checkout main
git pull origin main
git merge feature-xxx
git push origin main

# 4. 删除分支
git branch -d feature-xxx
git push origin --delete feature-xxx
```

---

## **📌 9. Git 常见问题**
| **问题** | **解决方法** |
|----------|------------|
| `git push` 报错 `rejected` | 先执行 `git pull origin main`，然后 `git merge main` 解决冲突 |
| `fatal: not a git repository` | 确保在正确的项目目录，或使用 `git init` 重新初始化 |
| `Merge conflict`（合并冲突） | 使用 VSCode 手动修改冲突文件，然后 `git add . && git commit` |
| **误删分支** | 使用 `git reflog` 找回丢失的提交 |

---

## **📌 10. 代码管理规则**
- `main` **必须保持稳定**，不能直接开发，必须通过 **分支合并** 进行修改。
- **每个功能独立分支**，不要在 `main` 直接开发。
- 提交代码时，**确保写清楚 commit 信息**（例如 `"完成 NLP 查询 API"`）。
- **合并前请先 `pull main`，避免冲突**。

---

**🚀 这样，每个团队成员都能轻松协作，避免 Git 版本冲突，提高开发效率！💡**