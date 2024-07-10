% Bias file for Michalski's trains problem

% Mode declarations
modeh(eastbound(train)).
modeb(short(car)).
modeb(closed(car)).
modeb(long(car)).
modeb(open_car(car)).
modeb(double(car)).
modeb(jagged(car)).
modeb(shape(car,shape)).
modeb(load(car,shape,int)).
modeb(wheels(car,int)).
modeb(has_car(train,car)).

% Determinations
determination(eastbound/1,short/1).
determination(eastbound/1,closed/1).
determination(eastbound/1,long/1).
determination(eastbound/1,open_car/1).
determination(eastbound/1,double/1).
determination(eastbound/1,jagged/1).
determination(eastbound/1,shape/2).
determination(eastbound/1,load/3).
determination(eastbound/1,wheels/2).
determination(eastbound/1,has_car/2).
