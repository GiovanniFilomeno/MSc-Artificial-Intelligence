enable_recursion.

head_pred(max,2).
body_pred(head,2).
body_pred(tail,2).
body_pred(empty,1).
body_pred(greater,2).

type(max,(list,element)).
type(head,(list,element)).
type(tail,(list,list)).
type(empty,(list,)).
type(greater,(element,element)).

direction(max,(in,out)).
direction(head,(in,out)).
direction(tail,(in,out)).
direction(empty,(in,)).
direction(greater,(in,in)).