"""The HTTP edge: one module per resource.

Routers only authenticate, validate against `app.schemas` and delegate to `app.agent`.
Nothing here contains business logic.
"""
