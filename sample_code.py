def add(a, b):
    """Add two numbers."""
    return a + b



def complex_fn(x):
    for i in range(5):
        if i % 2 == 0:
            x += i
    return x