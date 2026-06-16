max_assignment([4,2,6], 6).
max_assignment([1,1,1], 1).
max_assignment([1,3,2], 3).
max_assignment([3,4,5], 5).
max_assignment([3,5], 5).
max_assignment([10], 10).
max_assignment([10,10,5], 10).
max_assignment([7,3,9,2], 9).
max_assignment([-1, -3, -2, -1], -1).   % Negative numbers
max_assignment([0, -1, -2, -3], 0).      % Mix of negative and zero
max_assignment([50, 20, 30, 40], 50).    % Larger numbers
max_assignment([2, 5, 3, 5, 2], 5).      % Maximum appears multiple times
max_assignment([12, 15, 14, 13], 15).    % Max in the middle
max_assignment([21], 21).                % Single element
max_assignment([99, 101, 100], 101).     % Three elements, max in middle
max_assignment([6, 6, 6, 6], 6).         % All elements are max
max_assignment([-5, 1, -1, 0], 1).       % Positive and negative values
max_assignment([100, 200, 300, 400, 500], 500). % Ascending order
max_assignment([500, 400, 300, 200, 100], 500). % Descending but max at start
max_assignment([8, 10, 12, 14, 16, 18, 20, 19, 17, 15, 13, 11, 9], 20). % Long list, max at middle
max_assignment([45, 45, 45, 46, 45], 46).  % Nearly all the same except max
max_assignment([2, 8, 2, 8, 2, 8, 9], 9).  % Repeated numbers with one max
max_assignment([1, 1000, 100, 10], 1000).  % Significant range difference
