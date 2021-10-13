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
docsify serve docs
fi