message=$1

rm -rf docs/
cp -R /Users/fangzeyu/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/我的笔记 docs/
rm -rf docs/.obsidian
rm -rf docs/Papers
rm -rf docs/Projects
rm -rf docs/Day\ Planners

# 复制 README.md
cp docs/README.md README.md

# 更新 master
git add .
git commit -m "$message"
git push