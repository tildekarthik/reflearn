import re
import string
import itertools

def valid(expr):
    "Returns if an expression is valid or not"
    try:
        return not re.search(r'\b0[0-9]',expr) and eval(expr) is True
    except ZeroDivisionError:
        return False

def solve(formula):
    fillins = fill_in(formula)
    for f in fillins:
        if valid(f):
            return f     

def fill_in(formula):
    "Returns all possible values filled in for the formula"
    chars = ''.join(set(re.findall('[A-Z]', formula)))
    for digits in itertools.permutations('1234567890',len(chars)):
        tbl = str.maketrans(chars, ''.join(digits))
        yield formula.translate(tbl)

print(solve('ODD + ODD == EVEN'))
