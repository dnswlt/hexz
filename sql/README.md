# PostgreSQL

Hexz can optionally connect to a PostgreSQL database to store game history.
This enables the /history URL endpoint, which displays recently played
games that users can view.

The following assumes that you have installed a PostgreSQL database server,
use peer authentication, and have access to a terminal on that host.

Create a database `hexz`:

```bash
sudo -i -u postgres

psql <<EOD
CREATE ROLE hexz LOGIN PASSWORD 'your_password';

CREATE DATABASE hexz
    WITH
    OWNER = hexz
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8';
EOD
```

For development, do the same for a test database `hexz_test` that you can use
for integration tests:

```bash
psql <<EOD
CREATE ROLE hexz_test LOGIN PASSWORD 'hexz_test';

CREATE DATABASE hexz_test
    WITH
    OWNER = hexz_test
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8';
EOD
```

For each database, create the relevant tables:

```bash
psql -f schema.sql -h $HOSTNAME hexz hexz 
psql -f schema.sql -h $HOSTNAME hexz_test hexz_test
```
