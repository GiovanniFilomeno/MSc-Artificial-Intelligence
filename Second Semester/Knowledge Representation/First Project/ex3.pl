% Helper predicate to handle the compression of sequences
compress_helper([], [], _Start, _End, _LastAdded).
compress_helper([H|T], Compressed, Start, End, LastAdded) :-
    Next is End + 1,
    (   H == Next
    ->  % Continue the current sequence
        compress_helper(T, Compressed, Start, H, LastAdded)
    ;   % Sequence breaks, decide on compression
        Length is End - Start + 1,
        (   Length > 2
        ->  NewSegment = [[Start, End]]
        ;   Length == 2
        ->  NewSegment = [Start, End]
        ;   NewSegment = [Start]
        ),
        % Recursive call with new start, reset end
        compress_helper(T, Rest, H, H, H),
        % Append the new segment to the result
        append(NewSegment, Rest, Compressed)
    ).


% Public interface for compression
compress([H|T], Compressed) :-
    compress_helper(T, Compressed, H, H, H).
compress([], []).

% Initialization for an empty list compression
compress([], []).

% Helper predicate for expanding range [Start, End]
expand_range(Start, End, List) :-
    findall(X, between(Start, End, X), List).

% Decompressing the compressed list
decompress_helper([], []).
decompress_helper([H|T], List) :-
    (   is_list(H)
    ->  H = [Start, End],
        expand_range(Start, End, Expanded),
        decompress_helper(T, Rest),
        append(Expanded, Rest, List)
    ;   decompress_helper(T, Rest),
        List = [H|Rest]
    ).

% Public interface for decompression
decompress(Compressed, List) :-
    decompress_helper(Compressed, List).

% Test cases
:- writeln('Testing compression:'),
   compress([1,2,4,5,6,7,8,10,15,16,17,20,21,22,23,24,25], CompressedList),
   writeln(CompressedList).

:- writeln('Testing decompression:'),
   decompress([1, 2, [4, 8], 10, [15, 17], [20, 25]], DecompressedList),
   writeln(DecompressedList).
