% Operator definitions
op(add, X, Y, Z) :- Z is X + Y.
op(sub, X, Y, Z) :- Z is X - Y.
op(mul, X, Y, Z) :- Z is X * Y.
op(div, X, Y, Z) :- Y \= 0, Z is X / Y.
