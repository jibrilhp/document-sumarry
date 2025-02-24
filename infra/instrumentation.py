from typing import Callable
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Counter
from contextvars import ContextVar

request_tokens_var: ContextVar[int] = ContextVar("request_tokens", default=0)
response_tokens_var: ContextVar[int] = ContextVar("response_tokens", default=0)

def request_token_counter() -> Callable[[Info], None]:
    METRIC = Counter(
        "llm_request_token_counter",
        "number of token spent to process request",
        labelnames=("request_llm_token")
    )

    def instrumentation(info: Info):
        try:
            request_token_count = request_tokens_var.get()
            print(request_token_count)
            METRIC.inc(request_token_count)
        except:
            pass
    return instrumentation

def response_token_counter() -> Callable[[Info], None]:
    METRIC = Counter(
        "llm_response_token_counter",
        "number of token spent to process response",
        labelnames=("response_llm_token")
    )

    def instrumentation(info: Info):
        try:
            response_token_count = response_tokens_var.get()
            print(response_token_count)
            METRIC.inc(response_token_count)
        except:
            pass
    return instrumentation