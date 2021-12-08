message=$1

./cp.sh 

# 复制 README.md
cp docs/README.md README.md

# 更新 master
git add .
git commit -m "$message"
git push