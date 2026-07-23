# Security Policy

## Redis backend serialization

`RedisCacheBackend` serializes cached responses with [msgpack](https://msgpack.org/): a
cache entry is written as plain data (audio bytes, ints, floats, strings) and read back with
`CachedTTSResponse.from_msgpack`, which constructs only known dataclasses from those fields.
Reading an entry never instantiates arbitrary types or executes code, so a poisoned entry
cannot achieve remote code execution — the [CWE-502](https://cwe.mitre.org/data/definitions/502.html)
deserialization class of bug does not apply. An entry that is malformed or written by an
incompatible schema version fails to decode and is treated as a cache miss (and deleted to
self-heal).

Standard operational hygiene still applies — authenticate Redis (`requirepass`/ACLs) and
keep it network-isolated — but a shared or multi-tenant Redis no longer exposes this package
to code execution through the cache keyspace.

The in-memory `MemoryCacheBackend` stores live Python objects and does not deserialize
external bytes at all.

## Reporting a Vulnerability

Please email `omchauhan64408@gmail.com`.
