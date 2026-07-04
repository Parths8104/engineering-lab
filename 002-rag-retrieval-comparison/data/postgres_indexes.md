# PostgreSQL Index Types

## B-tree

The default index type. Supports equality and range queries on ordered data. If you write `CREATE INDEX ON users (email)` without specifying a type, you get a B-tree. Best for numeric, date, and short-string columns queried with `=`, `<`, `>`, or `BETWEEN`.

## Hash

Supports equality lookups only, no ordering. Hash indexes were considered unsafe until Postgres 10 (they weren't crash-safe); since then they're fully supported and can outperform B-tree for pure equality queries on large tables.

## GIN — Generalized Inverted Index

The right choice for indexing composite values: array elements, JSONB keys, or full-text search over `tsvector` columns. A GIN index on a JSONB column supports `@>`, `?`, and `?&` operators, letting you query nested keys without deserializing the whole document.

## GiST — Generalized Search Tree

Supports geometric data, ranges, and nearest-neighbor searches. The PostGIS spatial extension uses GiST internally. GiST also powers the `<->` distance operator on `point` columns for KNN queries.

## BRIN — Block Range INdex

Efficient for very large tables where the physical row order correlates with the indexed column — timestamps in append-only tables are the canonical example. BRIN indexes are dramatically smaller than B-tree at the cost of less precise lookups.
