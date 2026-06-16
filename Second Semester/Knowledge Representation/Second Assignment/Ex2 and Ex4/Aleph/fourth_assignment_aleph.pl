% Load Aleph
:- consult('aleph-swi.pl').

% Initialize Aleph
:- init(swi).

:- chdir('examples/fourth_assignment').
:- read_all(fourth_assignment).

:- induce.

:- (   fourth_assignment([3,4,2], 14, Ops),
       Ops \= []
   ->  write('Operations to transform [3,4,2] into 14: '), write(Ops), nl
   ;   write('No solution found to transform [3,4,2] into 14.'), nl
   ).
