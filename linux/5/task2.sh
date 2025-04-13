#!/bin/bash

read -p "Введите имя:" name
read -p "Введите фамилию:" surname

if [ "$name" = "Adam" ] && [ "$surname" = "Bond" ]; then
	echo "Access Granted"
fi
