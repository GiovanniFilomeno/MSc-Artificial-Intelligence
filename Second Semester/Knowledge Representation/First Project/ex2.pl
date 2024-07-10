% Define the predicate compress/2
compress(List, CompressedList) :-
    compress_helper(List, CompressedList, 1).

% Helper predicates for compression
compress_helper([], [], _).
compress_helper([X], [[X, Count]], Count) :- Count > 2.
compress_helper([X], [X, X], 2).
compress_helper([X], [X], 1).
compress_helper([X, Y | Rest], Compressed, Count) :-
    X \= Y,
    (   Count > 2
    ->  Compressed = [[X, Count] | RestCompressed]
    ;   Count = 2
    ->  Compressed = [X, X | RestCompressed]
    ;   Compressed = [X | RestCompressed]),
    compress_helper([Y | Rest], RestCompressed, 1).
compress_helper([X, X | Rest], Compressed, Count) :-
    NewCount is Count + 1,
    compress_helper([X | Rest], Compressed, NewCount).

% Define the predicate decompress/2
decompress(CompressedList, List) :-
    decompress_helper(CompressedList, List).

% Helper predicate for decompression
decompress_helper([], []).
decompress_helper([[N, C] | T], List) :-
    C > 2,
    length(Full, C),
    maplist(=(N), Full),
    decompress_helper(T, Rest),
    append(Full, Rest, List).
decompress_helper([X | T], [X | List]) :-
    decompress_helper(T, List).

% Test predicates
test_compress :-
    compress([2,2,2,3,3,3,4,4,5,5,5,5,6,6,6,6,6], CompressedList),
    writeln('Compressed:'), writeln(CompressedList).

test_decompress :-
    decompress([[2, 3], [3, 3], 4, 4, [5, 4], [6, 5]], DecompressedList),
    writeln('Decompressed:'), writeln(DecompressedList).

% Execute tests
:- initialization(test_compress).
:- initialization(test_decompress).
