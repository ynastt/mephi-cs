#1/bin/bash

echo "Введите имя пользователя"
read username
cat /etc/passwd | grep $username
