# FastAPI Dependency Injection

## The Depends() function

FastAPI's dependency injection is expressed through the `Depends()` marker. You declare a parameter as `param = Depends(callable)` and FastAPI calls that callable for every incoming request, passing its return value into your endpoint. Dependencies can themselves declare further dependencies, and FastAPI resolves the tree top-down.

## Per-request caching

By default, if the same dependency callable is declared by multiple dependencies in one request, FastAPI only calls it once and reuses the result. This is what makes `Depends(get_db)` safe as a shared session — you get the same session everywhere in the request. Set `use_cache=False` to disable this behavior when you need each caller to receive a fresh value.

## Yield-based dependencies

A dependency that uses `yield` can perform cleanup after the endpoint returns. Common pattern:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

The `finally` block runs after the response is sent, so it's the right place for connection cleanup, transaction commits, or cache invalidation.
