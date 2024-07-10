:- modeh(*,max_assignment(+list, #integer)).
:- modeb(*,max_assignment(+list, #integer)).
:- modeb(1, ((+list) = ([-integer | +list]))).
:- modeb(1, head(+list, #integer)).
:- modeb(1, tail(+list, -list)).

:- set(i,3).
:- set(noise,0).

:- determination(max_assignment/2, head/2).
:- determination(max_assignment/2, tail/2).
:- determination(max_assignment/2, max_assignment/2).
