# HTTP Status Codes

## 2xx — Success

- **200 OK** — Standard success response
- **201 Created** — A new resource was created; the response should include a Location header
- **204 No Content** — Success but nothing to return in the body

## 4xx — Client Errors

- **400 Bad Request** — Malformed request (invalid JSON, missing fields)
- **401 Unauthorized** — Missing or invalid credentials
- **403 Forbidden** — Credentials are valid but the caller lacks permission
- **404 Not Found** — The resource doesn't exist
- **409 Conflict** — Request conflicts with current state (e.g. duplicate creation)
- **422 Unprocessable Entity** — Well-formed request but semantically invalid; FastAPI returns this for Pydantic validation failures

## 5xx — Server Errors

- **500 Internal Server Error** — Generic server failure
- **502 Bad Gateway** — Upstream service returned an invalid response
- **503 Service Unavailable** — Server is temporarily overloaded or down for maintenance
- **504 Gateway Timeout** — Upstream service didn't respond in time
