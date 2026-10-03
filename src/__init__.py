"""Psyche package init.

COMPATIBILITY FIX (runs on any `import src...`, before any crew starts):
CrewAI >= 1.14 marks system/user messages with `cache_breakpoint: true` for
prompt caching, but only strips the flag for Anthropic. Groq's API rejects the
unknown field with:
    litellm.BadRequestError: ... property 'cache_breakpoint' is unsupported
CrewAI's executors import mark_cache_breakpoint at call time, so neutralizing
it here is effective. Safe to delete once CrewAI fixes it upstream.
See: https://github.com/crewaiinc/crewai/issues/5886
"""
try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda message: message  # noqa: E731
except Exception:  # very old CrewAI without crewai.llms.cache
    pass
