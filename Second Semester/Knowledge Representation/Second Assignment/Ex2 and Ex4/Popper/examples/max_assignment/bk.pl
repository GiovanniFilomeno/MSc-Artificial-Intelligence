tail([_|T], T).
head([H|_], H).
empty([]).

greater(X, Y) :- X > Y.
