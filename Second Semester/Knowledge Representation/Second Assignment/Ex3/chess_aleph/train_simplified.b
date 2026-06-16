% mode for the head
:- modeh(*, illegal(+pos, +pos, + pos, + pos, + pos, +pos)).

% encoding of illegal positions
:- modeb(*, adj(+pos, +pos)).
:- modeb(*, lt(+pos, +pos)).
:- modeb(*, not(lt(+pos, +pos))).


% determinations for illegal/6
:- determination(illegal/6, adj/2).
:- determination(illegal/6, lt/2).
:- determination(illegal/6, not/1). % add not in order to be able to use it in rules


:- set(clauselength, 6).  
:- set(minpos, 70).  % only consider rules that cover at least 70 examples


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
