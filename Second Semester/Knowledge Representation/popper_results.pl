:- dynamic hypothesis_popper/1.
hypothesis_popper(f(V0,V1):- zero(V1),empty(V0)).
hypothesis_popper(f(V0,V1):- tail(V0,V3),f(V3,V2),succ(V2,V1)).
