#!/bin/bash

get_user_info() {
	read -p "Введите имя пользователя:" username
	echo Информация о пользователе:
	cat /etc/passwd | grep $username
}

create_user() {
	read -p "Введите имя пользователя:" username
	if sudo useradd -m "$username"; then
		echo "Пользователь создан"
	else
		echo "Ошибка при создании пользователя"
	fi
}

remove_user() {
	read -p "Введите имя пользователя:" username
	if sudo userdel -r "$username"; then
		echo "Пользователь удален"
	else 
		echo "Ошибка при удалении пользователя"
	fi
}

function menu() {
	echo "МЕНЮ"
	while true; do
		echo "Выберите действие"
		select action in "Посмотреть информацию о пользователе" \
			"Создать пользователя" \
			"Удалить пользователя" \
			"Выйти"; do
			case $REPLY in
				1) get_user_info break ;;
				2) create_user break ;;
				3) remove_user break ;;
				4) return 0 ;;
				*) echo "Невозможный вариант" break;;
			esac
		done

	done
}

menu
	
