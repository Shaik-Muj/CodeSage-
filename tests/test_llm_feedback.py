from app.ai_engine.llm_feedback import critique_code_placeholder


def test_critique_returns_str():
    assert isinstance(critique_code_placeholder('x=1'), str)
