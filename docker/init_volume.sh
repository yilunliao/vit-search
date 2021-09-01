#!/bin/bash

if [ ! -f "/tmp/first_run" ]; then
	touch /tmp/first_run
	PUID=${PUID:-9999}
	chown -R $PUID /code
	useradd -s /bin/bash -u $PUID -o -c "" -m user
	usermod -a -G root user
        ln -s /code /home/user/FastAccess
fi
 
exec gosu user "$@"
