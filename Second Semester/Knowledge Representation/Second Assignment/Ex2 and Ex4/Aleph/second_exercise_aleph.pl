% Load Aleph
:- consult('aleph-swi.pl').

% Initialize Aleph
:- init(swi).

% Set working directory and load the sum definitions and examples
:- chdir('examples/sum_assignment').
:- read_all(sum_assignment).

% Induce rules for sum and test them
:- induce.
:- sum_assignment([3,4,5], X), nonvar(X), write('Sum Result: '), write(X), nl.


% Clear previous definitions, change directory, and load max definitions
:- clear.
:- chdir('../max_assignment').
:- read_all(max_assignment).

% % Induce rules for max and test them
:- induce.
:- max_assignment([3,4,6], X), nonvar(X), write('Max Result: '), write(X), nl.
