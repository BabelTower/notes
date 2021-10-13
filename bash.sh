echo "[0] 更新Github仓库"
echo "[1] 启动local serve"
read -p "Enter number : " func_num
if [ $func_num -eq 0 ];
then
echo 
read -p "Enter commit name : " commit_name
./push.sh $commit_name
fi