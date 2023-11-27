FROM postgres:14.3

ENV POSTGRES_PASSWORD pizzapizza
ENV POSTGRES_DB hexzdb

COPY schema.sql /docker-entrypoint-initdb.d/
