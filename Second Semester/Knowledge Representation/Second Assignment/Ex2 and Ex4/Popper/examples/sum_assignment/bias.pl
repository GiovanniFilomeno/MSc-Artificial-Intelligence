enable_recursion.
max_vars(5).

head_pred(sum,2).
body_pred(head,2).
body_pred(tail,2).
body_pred(add,3).

type(sum,(list,element)).
type(head,(list,element)).
type(tail,(list,list)).
type(add,(element,element,element)).

direction(sum,(in,out)).
direction(head,(in,out)).
direction(tail,(in,out)).
direction(add,(in,in,out)).
