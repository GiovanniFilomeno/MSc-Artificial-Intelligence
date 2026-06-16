tail([_|T], T).
head([H|_], H).
empty([]).

add(X, Y, Z) :- Z is X + Y.
