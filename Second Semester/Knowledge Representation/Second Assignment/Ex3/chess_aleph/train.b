% mode for the head
:- modeh(*, illegal(+colwk, +rowwk, +colwr, +rowwr, +colbk, +rowbk)).

% encoding of illegal positions
% adjacent columns
:- modeb(*, adj(+colwk, +colbk)).
:- modeb(*, adj(+colwr, +colbk)).

% adjacent rows
:- modeb(*, adj(+rowwk, +rowbk)).
:- modeb(*, adj(+rowwr, +rowbk)).

% columns less than 
:- modeb(*, lt(+colwk, +colbk)).
:- modeb(*, lt(+colwr, +colbk)).
:- modeb(*, lt(+colbk, +colwk)).
:- modeb(*, lt(+colbk, +colwr)).

% rows less than
:- modeb(*, lt(+rowwk, +rowbk)).
:- modeb(*, lt(+rowwr, +rowbk)).
:- modeb(*, lt(+rowbk, +rowwk)).
:- modeb(*, lt(+rowbk, +rowwr)).

% columns greater than
:- modeb(*, not(lt(+colwk, +colbk))).
:- modeb(*, not(lt(+colwr, +colbk))).
:- modeb(*, not(lt(+colbk, +colwk))).
:- modeb(*, not(lt(+colbk, +colwr))).

% rows greater than
:- modeb(*, not(lt(+rowwk, +rowbk))).
:- modeb(*, not(lt(+rowwr, +rowbk))).
:- modeb(*, not(lt(+rowbk, +rowwk))).
:- modeb(*, not(lt(+rowbk, +rowwr))).

% check for same positions of pieces
:- modeb(*, same(+colwk, +colwr)).
:- modeb(*, same(+rowwk, +rowwr)).
:- modeb(*, same(+colwk, +colwr)).
:- modeb(*, same(+rowwk, +rowwr)).
:- modeb(*, same(+colbk, +colwr)).
:- modeb(*, same(+rowwk, +rowbk)).

% check for same column or row for all 3 pieces (not sure why this is illegal though)
:- modeb(*, sameb(+colwk, +colwr, +colbk)).
:- modeb(*, sameb(+rowwk, +rowwr, +rowbk)).

% Determinations for illegal/6
:- determination(illegal/6, adj/2).
:- determination(illegal/6, lt/2).
:- determination(illegal/6, not/1). % add not in order to be able to use it in rules
:- determination(illegal/6, same/2). % add same
:- determination(illegal/6, sameb/3). % add same b for 3 inputs

%:- set(clauselength, 6). % get ALL 57 rules -> too specific 
:- set(minpos, 20).  % only consider rules that cover at least 20 examples

% definition of same for entries having the same value.
same(X,Y) :- number(X), number(Y), X == Y.
sameb(X,Y,Z) :- number(X), number(Y), number(Z), X == Y , Y == Z. 


% <(R/C, R/C)
lt(0,1).
lt(0,2).
lt(0,3).
lt(0,4).
lt(0,5).
lt(0,6).
lt(0,7).
lt(1,2).
lt(1,3).
lt(1,4).
lt(1,5).
lt(1,6).
lt(1,7).
lt(2,3).
lt(2,4).
lt(2,5).
lt(2,6).
lt(2,7).
lt(3,4).
lt(3,5).
lt(3,6).
lt(3,7).
lt(4,5).
lt(4,6).
lt(4,7).
lt(5,6).
lt(5,7).
lt(6,7).

% adj(R/C,R/C)
adj(0,0).
adj(1,1).
adj(2,2).
adj(3,3).
adj(4,4).
adj(5,5).
adj(6,6).
adj(7,7).
adj(0,1).
adj(1,2).
adj(2,3).
adj(3,4).
adj(4,5).
adj(5,6).
adj(6,7).
adj(7,6).
adj(6,5).
adj(5,4).
adj(4,3).
adj(3,2).
adj(2,1).
adj(1,0).
