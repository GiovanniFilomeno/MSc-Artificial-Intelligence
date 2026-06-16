% Background knowledge

% Definitions of pieces and their attributes
piece(p1, p11).
piece(p1, p12).
piece(p2, p21).
piece(p2, p22).
piece(p3, p31).
piece(p3, p32).
piece(p4, p41).
piece(p4, p42).
piece(p5, p51).
piece(p5, p52).
piece(p6, p61).
piece(p6, p62).
piece(p7, p71).
piece(p7, p72).
piece(p8, p81).
piece(p8, p82).

% Colors
red(p11).
blue(p12).
green(p21).
red(p22).
green(p31).
blue(p32).
red(p41).
green(p42).
blue(p51).
green(p52).
red(p61).
blue(p62).
green(p71).
blue(p72).
red(p81).
blue(p82).

% Sizes
small(p11).
small(p21).
small(p31).
small(p41).
small(p51).
small(p61).
small(p71).
small(p81).

% Orientations
upright(p11).
upright(p21).
upright(p31).
upright(p41).
upright(p51).
upright(p61).
upright(p71).
upright(p81).

% Coordinates
coord1(p11, 1).
coord2(p11, 2).
coord1(p12, 1).
coord2(p12, 2).
coord1(p21, 2).
coord2(p21, 3).
coord1(p22, 2).
coord2(p22, 3).
coord1(p31, 3).
coord2(p31, 4).
coord1(p32, 3).
coord2(p32, 4).
coord1(p41, 4).
coord2(p41, 5).
coord1(p42, 4).
coord2(p42, 5).
coord1(p51, 5).
coord2(p51, 6).
coord1(p52, 5).
coord2(p52, 6).
coord1(p61, 6).
coord2(p61, 7).
coord1(p62, 6).
coord2(p62, 7).
coord1(p71, 7).
coord2(p71, 8).
coord1(p72, 7).
coord2(p72, 8).
coord1(p81, 8).
coord2(p81, 9).
coord1(p82, 8).
coord2(p82, 9).

% Contacts
contact(p11, p12).
contact(p21, p22).
contact(p31, p32).
contact(p41, p42).
contact(p51, p52).
contact(p61, p62).
contact(p71, p72).
contact(p81, p82).

% Relations
lhs(p11).
lhs(p21).
lhs(p31).
lhs(p41).
lhs(p51).
lhs(p61).
lhs(p71).
lhs(p81).

rhs(p12).
rhs(p22).
rhs(p32).
rhs(p42).
rhs(p52).
rhs(p62).
rhs(p72).
rhs(p82).

% Shapes
shape(p11, square).
shape(p21, circle).
shape(p31, triangle).
shape(p41, hexagon).
shape(p51, rectangle).
shape(p61, pentagon).
shape(p71, ellipse).
shape(p81, star).
