echo " --------------------------------------------"
echo "|                                            |"
echo "|          [0] update Github repo            |"
echo "|          [1] run local serve               |"
echo "|                                            |"
echo " --------------------------------------------"

read -p "Enter number : " func_num

if [ $func_num -eq 0 ];
then
read -p "Enter commit name : " commit_name
./push.sh $commit_name
fi

if [ $func_num -eq 1 ];
then
rm -rf docs/
cp -R /Users/fangzeyu/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/我的笔记 docs/
rm -rf docs/.obsidian
rm -rf docs/Papers
rm -rf docs/Projects
rm -rf docs/Day\ Planners
docsify serve docs
fi