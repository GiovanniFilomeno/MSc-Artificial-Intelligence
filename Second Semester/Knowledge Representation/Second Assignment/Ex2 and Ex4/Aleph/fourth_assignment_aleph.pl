% Load Aleph
:- consult('/Users/giovannifilomeno/Desktop/Master-Artificial-Intelligence/Knowledge Representation/Second Assignment/Aleph/aleph-swi.pl').

% Initialize Aleph
:- init(swi).

:- chdir('/Users/giovannifilomeno/Desktop/Master-Artificial-Intelligence/Knowledge Representation/Second Assignment/Aleph/examples/fourth_assignment').
:- read_all(fourth_assignment).

:- induce.

:- (   fourth_assignment([3,4,2], 14, Ops),
       Ops \= []
   ->  write('Operations to transform [3,4,2] into 14: '), write(Ops), nl
   ;   write('No solution found to transform [3,4,2] into 14.'), nl
   ).