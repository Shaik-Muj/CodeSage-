import os


def greet(name):
    print(f"Hello {name}")


def unsafe():
    eval('2 + 2') # intentional security issue