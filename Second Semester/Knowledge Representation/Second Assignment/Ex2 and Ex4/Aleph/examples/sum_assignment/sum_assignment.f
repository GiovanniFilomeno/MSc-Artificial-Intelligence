sum_assignment([4,2,6], 12).
sum_assignment([1,1,1], 3).
sum_assignment([3,5], 8).
sum_assignment([], 0).
sum_assignment([10,10,5], 25).
sum_assignment([3,4,5], 12).
sum_assignment([1, 2, 3, 4, 5], 15).    % Increasing sequence
sum_assignment([-1, -2, -3], -6).        % Negative numbers
sum_assignment([-5, 5, 10, -10], 0).     % Summing to zero
sum_assignment([100, 200, 300], 600).    % Larger numbers
sum_assignment([0, 0, 0, 0], 0).         % All zeros
sum_assignment([1, -1, 1, -1, 2], 2).    % Alternating sign
sum_assignment([15], 15).                % Single element
sum_assignment([7, 8, -15, 20], 20).     % Mix of positive and negative
sum_assignment([5, 5, 5, 5, 5, 5], 30).  % Repeated numbers
sum_assignment([10, -5, -5], 0).         % Cancelling out to zero
sum_assignment([7, 0, 8, 0], 15).        % Mix of non-zero and zero
sum_assignment([12, 18, 25, 20], 75).    % Larger sums
