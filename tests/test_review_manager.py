from app.core.review_manager import review_source


def test_review_source():
    res = review_source('def a():\n    pass')
    assert 'critique' in res
