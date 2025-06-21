# GitHub Publishing Instructions

Follow these steps to publish your Fake News Detection project on your GitHub account:

## 1. Prepare Your Repository

1. Check that the README.md file reflects your project accurately

2. Review the license file:
   - Open `LICENSE` and make sure your name is in the copyright notice

3. Review the .gitignore file:
   - The .gitignore file is already set up to exclude common files and directories
   - It ignores Python bytecode, virtual environments, IDE files, logs, etc.

## 2. Initialize Git Repository

Open your terminal in the project directory and run:

```
git init
```

## 3. Add Your Files

```
git add .
```

## 4. Make Your First Commit

```
git commit -m "Initial commit: Fake News Detection System"
```

## 5. Create a GitHub Repository

1. Go to [GitHub](https://github.com/)
2. Click on the "+" icon in the top-right corner
3. Select "New repository"
4. Enter "fake-news-detection" as the repository name
5. Add a short description
6. Keep it as a public repository
7. Do NOT initialize with README, .gitignore, or license (we've created these already in our local repository)
8. Click "Create repository"

## 6. Link and Push to GitHub

GitHub will show commands to push an existing repository. Run the commands shown, which should be similar to:

```
git remote add origin https://github.com/YOUR-USERNAME/fake-news-detection.git
git branch -M main
git push -u origin main
```

## 7. Verify Your Repository

Visit your GitHub profile to ensure all files were uploaded correctly.

## 8. Removing Files From GitHub Repository

If you need to remove a file that was accidentally pushed (like README_GITHUB.md):

1. Delete the file locally:
   ```
   git rm README_GITHUB.md
   ```

2. Commit the removal:
   ```
   git commit -m "Remove README_GITHUB.md"
   ```

3. Push the changes:
   ```
   git push origin main
   ```

## 9. Updating Files on GitHub

When you make changes to files like README.md locally and want to update them on GitHub:

1. Stage the changes:
   ```
   git add README.md
   ```
   (Use `git add .` to stage all changes at once)

2. Commit the changes:
   ```
   git commit -m "Update README.md: Remove acknowledgments and fix duplicate structure sections"
   ```

3. Push the changes to GitHub:
   ```
   git push origin main
   ```

Your changes will now be reflected in the GitHub repository.

## 10. Set Up GitHub Pages (Optional)

If you want to create a project website:

1. Go to your repository settings
2. Scroll down to "GitHub Pages"
3. Select "main" as the source branch
4. Choose "/docs" as the folder (you'll need to create this folder with HTML files)
5. Click "Save"

---

Remember to never commit sensitive information such as API keys, passwords, or personal data to GitHub.
