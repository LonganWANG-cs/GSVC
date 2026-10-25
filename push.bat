@echo off
setlocal

for /f "delims=" %%B in ('git branch --show-current') do set "BRANCH=%%B"
if not defined BRANCH (
    echo Error: unable to determine the current branch. You may be in detached HEAD state.
    exit /b 1
)

echo Current branch: %BRANCH%
git add .
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "new" || exit /b 1
) else (
    echo No changes to commit.
)

git push -u origin "%BRANCH%"
