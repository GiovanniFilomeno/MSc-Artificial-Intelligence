fourth_assignment([2, 3], 5, [[add, 2, 3, 5]]).
fourth_assignment([1, 2, 3], 6, [[add, 1, 2, 3], [add, 3, 3, 6]]).
fourth_assignment([4, 5, 6, 7], 22, [[add, 4, 5, 9], [add, 9, 6, 15], [add, 15, 7, 22]]).

fourth_assignment([5, 3], 2, [[sub, 5, 3, 2]]).
fourth_assignment([10, 3, 2], 5, [[sub, 10, 3, 7], [sub, 7, 2, 5]]).
fourth_assignment([20, 5, 3, 2], 10, [[sub, 20, 5, 15], [sub, 15, 3, 12], [sub, 12, 2, 10]]).

fourth_assignment([2, 3], 6, [[mul, 2, 3, 6]]).
fourth_assignment([2, 3, 4], 24, [[mul, 2, 3, 6], [mul, 6, 4, 24]]).
fourth_assignment([2, 2, 2, 2], 16, [[mul, 2, 2, 4], [mul, 4, 2, 8], [mul, 8, 2, 16]]).

fourth_assignment([10, 2], 5, [[div, 10, 2, 5]]).
fourth_assignment([100, 5, 2], 10, [[div, 100, 5, 20], [div, 20, 2, 10]]).
fourth_assignment([64, 4, 2, 2], 8, [[div, 64, 4, 16], [div, 16, 2, 8], [div, 8, 1, 8]]).

fourth_assignment([2, 3, 4], 20, [[add, 2, 3, 5], [mul, 5, 4, 20]]).
fourth_assignment([1, 2, 3, 4], 24, [[add, 1, 2, 3], [mul, 3, 4, 12], [mul, 12, 2, 24]]).
fourth_assignment([6, 2, 3], 48, [[add, 6, 2, 8], [mul, 8, 3, 24], [mul, 24, 2, 48]]).

fourth_assignment([4, 5, 6], 35, [[mul, 4, 5, 20], [add, 20, 6, 26], [add, 26, 9, 35]]).
fourth_assignment([2, 3, 2, 4], 17, [[mul, 2, 3, 6], [mul, 6, 2, 12], [add, 12, 4, 16], [add, 16, 1, 17]]).

fourth_assignment([64, 4, 10], 26, [[div, 64, 4, 16], [add, 16, 10, 26]]).

fourth_assignment([1, 1, 1, 1, 1], 5, [[add, 1, 1, 2], [add, 2, 1, 3], [add, 3, 1, 4], [add, 4, 1, 5]]).
fourth_assignment([7, 3, 2], 27, [[add, 7, 3, 10], [mul, 10, 2, 20], [add, 20, 7, 27]]).
fourth_assignment([10, 5, 7, 3], 50, [[add, 10, 5, 15], [mul, 15, 3, 45], [add, 45, 5, 50]]).

fourth_assignment([30, 10, 5], 10, [[sub, 30, 10, 20], [sub, 20, 5, 15], [sub, 15, 5, 10]]).
fourth_assignment([50, 25, 10], 5, [[div, 50, 25, 2], [mul, 2, 10, 20], [sub, 20, 15, 5]]).
fourth_assignment([100, 50, 25], 0, [[sub, 100, 50, 50], [sub, 50, 25, 25], [sub, 25, 25, 0]]).

fourth_assignment([3, 3, 3, 3], 81, [[mul, 3, 3, 9], [mul, 9, 3, 27], [mul, 27, 3, 81]]).
fourth_assignment([1, 2, 3, 4, 5], 120, [[mul, 1, 2, 2], [mul, 2, 3, 6], [mul, 6, 4, 24], [mul, 24, 5, 120]]).
fourth_assignment([6, 7, 2, 3], 504, [[mul, 6, 7, 42], [mul, 42, 2, 84], [mul, 84, 3, 252], [mul, 252, 2, 504]]).

fourth_assignment([128, 4, 2, 2], 8, [[div, 128, 4, 32], [div, 32, 2, 16], [div, 16, 2, 8]]).
fourth_assignment([1000, 10, 5, 2], 10, [[div, 1000, 10, 100], [div, 100, 5, 20], [div, 20, 2, 10]]).
fourth_assignment([256, 2, 2, 2, 2], 4, [[div, 256, 2, 128], [div, 128, 2, 64], [div, 64, 2, 32], [div, 32, 2, 16], [div, 16, 2, 8], [div, 8, 2, 4]]).

fourth_assignment([15, 5, 3], 45, [[div, 15, 5, 3], [mul, 3, 15, 45]]).
fourth_assignment([9, 3, 2, 4], 40, [[div, 9, 3, 3], [mul, 3, 2, 6], [add, 6, 4, 10], [mul, 10, 4, 40]]).
fourth_assignment([16, 4, 2, 8], 32, [[div, 16, 4, 4], [mul, 4, 2, 8], [add, 8, 8, 16], [mul, 16, 2, 32]]).

fourth_assignment([5, 2, 4, 8, 3], 58, [[mul, 5, 2, 10], [div, 10, 2, 5], [mul, 5, 4, 20], [add, 20, 8, 28], [mul, 28, 2, 56], [add, 56, 2, 58]]).
fourth_assignment([8, 2, 4, 2, 10], 80, [[mul, 8, 2, 16], [div, 16, 4, 4], [mul, 4, 2, 8], [mul, 8, 10, 80]]).



