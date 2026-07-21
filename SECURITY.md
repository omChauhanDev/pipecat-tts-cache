# Security Policy

## Redis backend trust boundary

`RedisCacheBackend` serializes cached responses with Python `pickle` and calls
`pickle.loads` on data read back from Redis. Deserializing untrusted pickle data can lead
to arbitrary code execution (CWE-502).

**Therefore the Redis instance backing the cache must be trusted:** single-tenant,
authenticated (`requirepass`/ACLs), and network-isolated (not exposed to other tenants or
applications that could write to the keyspace). Do not point the Redis backend at a shared
or publicly reachable Redis.

The in-memory `MemoryCacheBackend` is not affected (it stores live Python objects and does
not deserialize external bytes).

## Reporting a Vulnerability

Please email `omchauhan64408@gmail.com`.
