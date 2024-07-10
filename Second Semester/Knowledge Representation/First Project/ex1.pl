% Operator definitions with evaluation
apply_op(add, X, Y, Z) :- Z is X + Y.
apply_op(sub, X, Y, Z) :- Z is X - Y.
apply_op(mul, X, Y, Z) :- Z is X * Y.
apply_op(div, X, Y, Z) :- Y \= 0, Z is X / Y.

% solve/3 - main predicate
solve(L, N, Ops) :-
    solve_helper(L, N, [], Ops).

% solve_helper/4 - helper predicate
solve_helper([N], N, Ops, Ops).
solve_helper([X,Y|Rest], N, AccOps, Ops) :-
    apply_op(Op, X, Y, Z),
    append(AccOps, [[0,Op]], NewAccOps),
    solve_helper([Z|Rest], N, NewAccOps, Ops).

% Example tests
test_solve :-
    solve([8,2,3,6,2], 27, Ops1), writeln(Ops1).
    % solve([1, 2, 3, 4], 10, Ops2), writeln(Ops2),
    % solve([5, 2, 4, 8], 1, Ops3), writeln(Ops3),
    % solve([10, 2, 5, 2], 40, Ops4), writeln(Ops4).

% Query to run tests
:- test_solve.
