message=$1

# 复制 README.md
cp README.md docs/README.md

# 更新 master
git add .
git commit -m "$message"
git push origin master