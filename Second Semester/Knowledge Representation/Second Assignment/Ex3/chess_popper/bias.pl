max_vars(6). % max number of distinct variables
max_body(4). % max number of body predicates 
max_clauses(4). % max number of clauses

head_pred(illegal, 6). % target predicate

% body predicates
body_pred(adj, 2).
body_pred(lt, 2). % less than
body_pred(gt, 2). % greater than - not(lt/2))
body_pred(same, 2). % same position
body_pred(sameb, 3). % same row or column

% number of arguments each predicate takes
type(illegal, (pos, pos, pos, pos, pos, pos)).
type(adj, (pos, pos)).
type(lt, (pos, pos)).
type(gt, (pos, pos)).
type(same, (pos, pos)).
type(sameb, (pos, pos, pos)).

% all arguments are inputs
direction(illegal, (in, in, in, in, in, in)).
direction(adj, (in, in)).
direction(lt, (in, in)).
direction(gt, (in, in)).
direction(same, (in, in)).
direction(sameb, (in, in, in)).
