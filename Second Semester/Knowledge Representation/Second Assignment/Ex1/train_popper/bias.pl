% Head Predicate
head_pred(eastbound,1).

% Body Predicates
body_pred(has_car,2).
body_pred(short,1).
body_pred(long,1).
body_pred(closed,1).
body_pred(open_car,1).
body_pred(double,1).
body_pred(jagged,1).
body_pred(shape,2).
body_pred(load,3).
body_pred(wheels,2).

% Types
type(eastbound,(train,)).
type(has_car,(train,car)).
type(short,(car,)).
type(long,(car,)).
type(closed,(car,)).
type(open_car,(car,)).
type(double,(car,)).
type(jagged,(car,)).
type(shape,(car,shape)).
type(load,(car,shape,num)).
type(wheels,(car,num)).


